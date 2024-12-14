import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import VICREGLoss
from impl import JEPA
from dataset import create_wall_dataloader
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model():

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    state_latent_dim = 256
    action_latent_dim = 32
    hidden_dim = 512

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
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,           # Learning rate
        momentum=0.9,      # Momentum term
        weight_decay=1e-4  # Regularization for weights
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # Define loss function
    vicreg_loss = VICREGLoss(
        lambda_invariance=1.0,   # Balances reconstruction fidelity
        lambda_variance=25.0,    # Strong regularization against collapse
        lambda_covariance=1.0    # Decorrelation of embedding dimensions
    )

    # Training loop
    num_epochs = 100
    best_train_loss = float('inf')  # Track the best training loss
    checkpoint_path = 'model_weights.pth'


    for epoch in range(num_epochs):
        model.train()
        epoch_vicreg_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            states = batch.states.to(device)  # [B, 17, 2, 65, 65]
            actions = batch.actions.to(device)  # [B, 16, 2]
            #states = (states - states.mean(dim=[2, 3, 4], keepdim=True)) / states.std(dim=[2, 3, 4], keepdim=True)
            # Predict state embeddings for all timesteps
            predicted_states = model(states[:, 0, :, :, :], actions)  # [B, 17, 256]

            target_states = states

            # Encode target states
            target_latent_states = model.state_encoder(target_states.reshape(-1, *target_states.shape[2:]))  # [B*16, 256]
            target_latent_states = target_latent_states.view(states.shape[0], target_states.shape[1], -1)  # [B, 16, 256]


            std = predicted_states.std(dim=0)
            mean_std = std.mean()  # Average standard deviation across all dimensions
            min_std = std.min()    # Minimum standard deviation
            max_std = std.max()    # Maximum standard deviation

            #print(f"Mean Std: {mean_std:.4f}, Min Std: {min_std:.4f}, Max Std: {max_std:.4f}")
            #print(f"Mean Std: {mean_std:.4f}, Min Std: {min_std:.4f}, Max Std: {max_std:.4f}")

            # Flatten for VICRegLoss
            predicted_states_flat = predicted_states.reshape(-1, predicted_states.shape[-1])  # [B*16, 256]
            target_latent_states_flat = target_latent_states.reshape(-1, target_latent_states.shape[-1])  # [B*16, 256]

            # Compute loss
            total_loss, inv_loss, var_loss, cov_loss = vicreg_loss(predicted_states_flat, target_latent_states_flat)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            epoch_vicreg_loss += total_loss.item()
            progress_bar.set_postfix({
                'Total Loss': f"{total_loss.item():.4f}",
                'Invariance': f"{inv_loss.item():.4f}",
                'Variance': f"{var_loss.item():.4f}",
                'Covariance': f"{cov_loss.item():.4f}"
            })

        # Step the learning rate scheduler
        scheduler.step()
        avg_vicreg_loss = epoch_vicreg_loss / len(train_loader)
        print(f"Epoch {epoch+1} - VicReg Loss: {avg_vicreg_loss:.4f}")

        if avg_vicreg_loss < best_train_loss:
            best_train_loss = avg_vicreg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model weights saved at epoch {epoch+1}")

    # Save final model
    torch.save(model, "final_model.pth")
    print("Training completed. Final model saved.")

if __name__ == "__main__":
    train_model()
    # Instantiate the model architecture

    # Initialize models
    state_latent_dim = 256
    action_latent_dim = 32
    hidden_dim = 512

    model = JEPA(state_latent_dim=state_latent_dim, action_latent_dim=action_latent_dim, hidden_dim=hidden_dim).to(device)
    # Load the saved weights
    model.load_state_dict(torch.load('model_weights.pth'))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")


