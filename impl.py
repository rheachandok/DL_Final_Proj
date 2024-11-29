import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class EncoderNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(EncoderNetwork, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Compute the final flattened size after convolutions

        self.flatten_size = 128 * 9 * 9  # 128 channels with 9x9 feature maps (after 3 strides of 2 from 65x65)
        # Fully Connected Layer
        self.fc = nn.Linear(self.flatten_size, hidden_dim)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))  # [batch_size, 32, 33, 33]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 17, 17]
        x = F.relu(self.conv3(x))  # [batch_size, 128, 9, 9]
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 128 * 9 * 9]
        x = self.fc(x)             # Output latent representation [batch_size, hidden_dim]
        return x


class PredictorNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PredictorNetwork, self).__init__()
        # Define GRU layer with input size as hidden_dim + action_dim and output as hidden_dim
        self.gru = nn.GRU(input_size=hidden_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        # Linear layer to generate the next latent state from GRU output
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, action):
        # Concatenate the current latent state with the action along the last dimension
        x = torch.cat([state, action], dim=-1).unsqueeze(1)  # [batch_size, 1, hidden_dim + action_dim]
        # Pass through GRU
        x, _ = self.gru(x)  # Output size: [batch_size, 1, hidden_dim]
        # Remove the sequence dimension and pass through fully connected layer
        x = self.fc(x.squeeze(1))  # Output size: [batch_size, hidden_dim]
        return x

class JEPA(nn.Module):

    def __init__(self, input_channels, hidden_dim, action_dim):
        super(JEPA, self).__init__()
        self.encoder = EncoderNetwork(input_channels, hidden_dim)
        self.predictor = PredictorNetwork(hidden_dim, action_dim)
        # Define the target encoder network (for BYOL)
        self.target_encoder = EncoderNetwork(input_channels, hidden_dim)
        # Initialize target encoder parameters
        self._initialize_target_encoder()
        # Set repr_dim to hidden_dim
        self.repr_dim = hidden_dim

    def _initialize_target_encoder(self):
        # Initialize target encoder parameters with encoder parameters
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(param.data)

    def update_target_encoder(self, momentum=0.99):
        # Update target encoder parameters
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data

    def forward(self, states, actions, return_targets=False):
        batch_size, seq_len, channels, height, width = states.size()  # Get the shape of the input states
        predicted_states = []
        target_states = []

        # Step 1: Encode the Initial Observation
        # Encode the first state to get the initial latent state
        s_t = self.encoder(states[:, 0])  # Encode the first time step
        predicted_states.append(s_t.unsqueeze(1))  # Add the latent state to predicted_states (unsqueeze to add time dimension)

        if return_targets:
            s_t_target = self.target_encoder(states[:, 0])
            target_states.append(s_t_target.unsqueeze(1))

        # Step 2: Recurrently Predict Future Latent States
        for t in range(1, seq_len):
            action_t = actions[:, t - 1]  # Get the action taken at time step t-1
            s_t = self.predictor(s_t, action_t)  # Predict the next latent state
            predicted_states.append(s_t.unsqueeze(1))  # Add the predicted latent state to the list

            if return_targets:
                s_t_target = self.target_encoder(states[:, t])
                target_states.append(s_t_target.unsqueeze(1))

        # Step 3: Concatenate Predicted States Across Time
        predicted_states = torch.cat(predicted_states, dim=1)  # Concatenate along the time dimension

        if return_targets:
            target_states = torch.cat(target_states, dim=1)
            return predicted_states, target_states
        else:
            return predicted_states

def compute_loss(predicted_states, target_states):
    # Normalize the representations
    predicted_states = F.normalize(predicted_states, dim=-1)
    target_states = F.normalize(target_states.detach(), dim=-1)  # Stop-gradient on target

    # Compute MSE loss
    loss = F.mse_loss(predicted_states, target_states)
    return loss

def train_model(model, dataloader, optimizer, num_epochs=10, momentum=0.99, device='cuda'):
    """
    Trains the JEPA model using the BYOL framework with tqdm progress bars.

    Args:
        model: The JEPA model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer for updating the model parameters.
        num_epochs (int): Number of training epochs.
        momentum (float): Momentum coefficient for updating the target encoder.
        device (str): Device to run the training on ('cuda' or 'cpu').

    Returns:
        None
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Initialize tqdm progress bar for batches
        with tqdm(total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit='batch') as pbar:
            for states, locations, actions in dataloader:
                states = states.to(device)  # Shape: [batch_size, seq_len, channels, height, width]
                actions = actions.to(device)  # Shape: [batch_size, seq_len - 1, action_dim]

                optimizer.zero_grad()

                # Forward pass with return_targets=True to get both predicted and target states
                predicted_states, target_states = model(states, actions, return_targets=True)
                # predicted_states and target_states shape: [batch_size, seq_len, hidden_dim]

                # Flatten the representations to combine batch and sequence dimensions
                batch_size, seq_len, hidden_dim = predicted_states.size()
                predicted_states_flat = predicted_states.view(batch_size * seq_len, hidden_dim)
                target_states_flat = target_states.view(batch_size * seq_len, hidden_dim)

                # Compute loss across all time steps and batch instances
                loss = compute_loss(predicted_states_flat, target_states_flat)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update the target encoder using exponential moving average
                model.update_target_encoder(momentum)

                total_loss += loss.item()

                # Update tqdm progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                pbar.update(1)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
