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
normalizer.compute_embedding_stats(model, train_loader, device)
print("Embedding Mean:", normalizer.mean)
print("Embedding Std:", normalizer.std)

# Define Optimizers and Scheduler
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler_model = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=5, gamma=0.1)

# Define loss functions
vicreg_loss = VICRegLoss(
    lambda_invariance=1.0,    # Hyperparameters to be tuned
    lambda_variance=10.0,
    lambda_covariance=0.1
)

# Training loop
num_epochs = 10
best_train_loss = float('inf')  # Track the best training loss
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

        # Extract target states for comparison (all states except the first one)
        target_states = states[:, 1:, :, :, :]  # [B, 16, 2, 65, 65] (all states except the first one)

        # Forward pass through JEPA model to get predicted state embeddings
        predicted_states = model(states, actions)  # [B, 16, state_latent_dim]

        # Encode target states into latent space
        target_latent_states = model.state_encoder(target_states.view(-1, *target_states.shape[2:]))  # [B*16, state_latent_dim]
        target_latent_states = target_latent_states.view(states.shape[0], target_states.shape[1], -1)  # [B, 16, state_latent_dim]

        # Normalize predicted stat
        # Encode target states into latent space
        target_latent_states = model.state_encoder(target_states.view(-1, *target_states.shape[2:]))  # [B*16, state_latent_dim]
        target_latent_states = target_latent_states.view(states.shape[0], target_states.shape[1], -1)  # [B, 16, state_latent_dim]

        # Normalize predicted states and target states if needed (optional)
        # predicted_states = normalizer.normalize_embeddings(predicted_states)
        # target_latent_states = normalizer.normalize_embeddings(target_latent_states)

        # Compute the loss for each predicted state vs corresponding target state in latent space
        total_loss = 0.0
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        for t in range(target_latent_states.shape[1]):
            total, inv, var, cov = vicreg_loss(predicted_states[:, t], target_latent_states[:, t])
            total_loss += total
            inv_loss += inv
            var_loss += var
            cov_loss += cov

        # Backpropagation for JEPA model
        optimizer_model.zero_grad()
        total_loss.backward()

        # Gradient Clipping (optional, can be enabled for stability)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Save the model if the training loss improves
        if avg_vicreg_loss < best_train_loss:
            best_train_loss = avg_vicreg_loss
            torch.save(model.state_dict(), checkpoint_path)  # Save only the model state_dict

            print(f"Model checkpoint saved at epoch {epoch+1}")

    # Optionally, after training, you can save the final model as well
    torch.save(model.state_dict(), "final_model.pth")
