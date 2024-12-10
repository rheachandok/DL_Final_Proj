import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import VICRegLoss
from impl import JEPA
from dataset import create_wall_dataloader
from normalization import Normalizer
from torch.optim.lr_scheduler import CosineAnnealingLR

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
state_latent_dim = 256
action_latent_dim = 32
hidden_dim = 256

model = JEPA(state_latent_dim=state_latent_dim, action_latent_dim=action_latent_dim, hidden_dim=hidden_dim).to(device)

# Create actual DataLoaders without normalization (handled in training loop)
train_loader = create_wall_dataloader(
    data_path="/scratch/DL24FA/train",
    device=device,
    batch_size=64,
    probing=False,  # Set to False since 'locations' are not used
    train=True,
)

# Initialize Normalizer
normalizer = Normalizer(device=device)
#normalizer.compute_embedding_stats(model, train_loader, device)
print("Embedding Mean:", normalizer.mean)
print("Embedding Std:", normalizer.std)

# Define Optimizers and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Define loss functions
vicreg_loss = VICRegLoss(
    lambda_invariance=1.0,    # Hyperparameters to be tuned
    lambda_variance=10.0,
    lambda_covariance=0.1
)

# Training loop
num_epochs = 100
best_train_loss = float('inf')  # Track the best training loss
checkpoint_path = 'best_model.pth'

def normalize_embeddings(embeddings):
    return F.normalize(embeddings, p=2, dim=-1)

for epoch in range(num_epochs):
    model.train()
    epoch_vicreg_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch in progress_bar:
        states = batch.states.to(device)
        actions = batch.actions.to(device)
        # Extract target states for comparison (all states except the first one)
        target_states = states[:, 1:, :, :, :]  # [B, 16, 2, 65, 65] (all states except the first one)

        predicted_states = model(states, actions)
        # Encode target states into latent space
        predicted_states = predicted_states[:, 1:, :]
        target_latent_states = model.state_encoder(target_states.reshape(-1, *target_states.shape[2:]))
        target_latent_states = target_latent_states.reshape(states.shape[0], target_states.shape[1], -1)

        predicted_states = normalize_embeddings(predicted_states)
        target_latent_states = normalize_embeddings(target_latent_states)

        print("Pred:",predicted_states.shape)
        print("Target:", target_latent_states.shape)

        predicted_states_flat = predicted_states.reshape(-1, predicted_states.shape[-1])  # [B*T, state_latent_dim]
        target_latent_states_flat = target_latent_states.reshape(-1, target_latent_states.shape[-1])  # [B*T, state_latent_dim]

        print(f"predicted_states_flat shape: {predicted_states_flat.shape}")
        print(f"target_latent_states_flat shape: {target_latent_states_flat.shape}")

        total_loss, inv_loss, var_loss, cov_loss = vicreg_loss(predicted_states_flat, target_latent_states_flat)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_vicreg_loss += total_loss.item()
        progress_bar.set_postfix({
            'Total Loss': f"{total_loss.item():.4f}",
            'Invariance': f"{inv_loss.item():.4f}",
            'Variance': f"{var_loss.item():.4f}",
            'Covariance': f"{cov_loss.item():.4f}",
            'Grad Norm': f"{torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0):.4f}"
        })

    scheduler.step()
    avg_vicreg_loss = epoch_vicreg_loss / len(train_loader)
    print(f"Epoch {epoch+1} - VICReg Loss: {avg_vicreg_loss:.4f}")

    if avg_vicreg_loss < best_train_loss:
        best_train_loss = avg_vicreg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_train_loss
        }, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch+1}")

torch.save(model.state_dict(), "final_model.pth")
