import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import VICRegLoss
from impl import JEPA
from dataset import create_wall_dataloader
from normalization import Normalizer

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
state_latent_dim = 128
action_latent_dim = 32
hidden_dim = 256

model = JEPA(state_latent_dim=state_latent_dim, action_latent_dim=action_latent_dim, hidden_dim=hidden_dim).to(device)

# Initialize Normalizer
normalizer = Normalizer()

# Function to compute normalization statistics for embeddings
def compute_embedding_stats(model, data_loader, normalizer, device):
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in tqdm(data_loader, desc="Computing Embedding Stats"):
            states = batch.states.to(device)    # [B,17,2,65,65]
            actions = batch.actions.to(device)  # [B,16,2]
            target_states = states[:, 16, :, :, :]  # [B, 2, 65, 65]
            Sy = model.state_encoder(target_states)  # [B, state_latent_dim]
            all_embeddings.append(Sy)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        normalizer.embedding_mean = all_embeddings.mean(dim=0)
        normalizer.embedding_std = all_embeddings.std(dim=0) + 1e-6  # Avoid division by zero


# Create actual DataLoaders without normalization (handled in training loop)
train_loader = create_wall_dataloader(
    data_path="/scratch/DL24FA/train",
    device=device,
    batch_size=64,
    probing=False,  # Set to False since 'locations' are not used
    train=True,
)

normalizer.compute_embedding_stats(model, train_loader, device)
print("Embedding Mean:", normalizer.mean)
print("Embedding Std:", normalizer.std)

# Since you mentioned not needing evaluation code, val_loader is omitted
# If needed, create similarly with probing=False

# Define Optimizers and Scheduler
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler_model = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=5, gamma=0.1)

# Define loss functions
vicreg_loss = VICRegLoss(
    lambda_invariance=25.0,    # Hyperparameters to be tuned
    lambda_variance=25.0,
    lambda_covariance=1.0
)

# Training loop
num_epochs = 10
best_train_loss = float('inf')  # Changed to track best training loss
checkpoint_path = 'best_model.pth'

for epoch in range(num_epochs):
    model.train()
    epoch_vicreg_loss = 0.0
    epoch_inv_loss = 0.0
    epoch_var_loss = 0.0
    epoch_cov_loss = 0.0

    # Initialize tqdm progress bar for the current epoch
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

    for batch in progress_bar:
        states = batch.states.to(device)    # [B,17,2,65,65]
        actions = batch.actions.to(device)  # [B,16,2]

        # Encode target Sy externally
        target_states = states[:, 16, :, :, :]  # [B, 2, 65, 65]
        with torch.no_grad():
            Sy = model.state_encoder(target_states)  # [B, state_latent_dim]
        Sy = normalizer.normalize_embeddings(Sy)      # Normalize targets

        # Forward pass through JEPA model to get Sy_hat
        Sy_hat = model(states, actions)             # [B, state_latent_dim]
        Sy_hat = normalizer.normalize_embeddings(Sy_hat)  # Normalize predictions

        # Compute VICReg-like total loss
        total_loss, inv_loss, var_loss, cov_loss = vicreg_loss(Sy_hat, Sy)

        # Backpropagation for JEPA model
        optimizer_model.zero_grad()
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer_model.step()

        # Accumulate losses for monitoring
        epoch_vicreg_loss += total_loss.item()
        epoch_inv_loss += inv_loss.item()
        epoch_var_loss += var_loss.item()
        epoch_cov_loss += cov_loss.item()

        # Update tqdm progress bar with current batch loss
        progress_bar.set_postfix({
            'Total Loss': f"{total_loss.item():.4f}",
            'Invariance Loss': f"{inv_loss.item():.4f}",
            'Variance Loss': f"{var_loss.item():.4f}",
            'Covariance Loss': f"{cov_loss.item():.4f}"
        })

    # Step the scheduler
    scheduler_model.step()

    # Calculate average losses for the epoch
    avg_vicreg_loss = epoch_vicreg_loss / len(train_loader)
    avg_inv_loss = epoch_inv_loss / len(train_loader)
    avg_var_loss = epoch_var_loss / len(train_loader)
    avg_cov_loss = epoch_cov_loss / len(train_loader)

    print(f"Epoch {epoch+1} - VICReg Loss: {avg_vicreg_loss:.4f}, "
          f"Invariance: {avg_inv_loss:.4f}, Variance: {avg_var_loss:.4f}, "
          f"Covariance: {avg_cov_loss:.4f}")

    # Checkpointing as previously defined...
 
    torch.save(model, "model.pth")
