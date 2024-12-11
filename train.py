import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import SLIMCRLoss
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

# Create DataLoader
train_loader = create_wall_dataloader(
    data_path="/scratch/DL24FA/train",
    device=device,
    batch_size=64,
    probing=False,  # 'locations' not used during training
    train=True,
)

# Define Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Define loss function
slimcr_loss = SLIMCRLoss(
    lambda_invariance=1.0,   # Balances reconstruction fidelity
    lambda_variance=15.0,    # Strong regularization against collapse
    lambda_covariance=1.0    # Decorrelation of embedding dimensions
)


# Training loop
num_epochs = 100
best_train_loss = float('inf')  # Track the best training loss
checkpoint_path = 'best_model.pth'


def normalize_embeddings(embeddings):
    """L2 normalize embeddings."""
    return F.normalize(embeddings, p=2, dim=-1)


for epoch in range(num_epochs):
    model.train()
    epoch_slimcr_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch in progress_bar:
        states = batch.states.to(device)  # [B, 17, 2, 65, 65]
        actions = batch.actions.to(device)  # [B, 16, 2]

        # Target states for comparison (all states except the first one)
        target_states = states[:, 1:, :, :, :]  # [B, 16, 2, 65, 65]

        # Predict state embeddings for all timesteps
        predicted_states = model(states[:, 0, :, :, :], actions)  # [B, 17, 256]
        predicted_states = predicted_states[:, 1:, :]  # Remove the first state embedding

        # Encode target states
        target_latent_states = model.state_encoder(target_states.reshape(-1, *target_states.shape[2:]))  # [B*16, 256]
        target_latent_states = target_latent_states.view(states.shape[0], target_states.shape[1], -1)  # [B, 16, 256]

        # Normalize embeddings
        predicted_states = normalize_embeddings(predicted_states)  # [B, 16, 256]
        target_latent_states = normalize_embeddings(target_latent_states)  # [B, 16, 256]

        # Flatten for SlimCR
        predicted_states_flat = predicted_states.reshape(-1, predicted_states.shape[-1])  # [B*16, 256]
        target_latent_states_flat = target_latent_states.reshape(-1, target_latent_states.shape[-1])  # [B*16, 256]

        # Compute loss
        total_loss, inv_loss, var_loss, cov_loss = slimcr_loss(predicted_states_flat, target_latent_states_flat)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update progress bar
        epoch_slimcr_loss += total_loss.item()
        progress_bar.set_postfix({
            'Total Loss': f"{total_loss.item():.4f}",
            'Invariance': f"{inv_loss.item():.4f}",
            'Variance': f"{var_loss.item():.4f}",
            'Covariance': f"{cov_loss.item():.4f}"
        })

    # Step the learning rate scheduler
    scheduler.step()
    avg_slimcr_loss = epoch_slimcr_loss / len(train_loader)
    print(f"Epoch {epoch+1} - SlimCR Loss: {avg_slimcr_loss:.4f}")

    # Save checkpoint if loss improves
    if avg_slimcr_loss < best_train_loss:
        best_train_loss = avg_slimcr_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_train_loss
        }, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch+1}")

# Save final model
torch.save(model, "final_model.pth")
print("Training completed. Final model saved.")