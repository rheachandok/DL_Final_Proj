import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, states, actions):
        batch_size, seq_len, channels, height, width = states.size()  # Get the shape of the input states
        predicted_states = []

        # Step 1: Encode the Initial Observation
        # Encode the first state to get the initial latent state
        s_t = self.encoder(states[:, 0])  # Encode the first time step
        predicted_states.append(s_t.unsqueeze(1))  # Add the latent state to predicted_states (unsqueeze to add time dimension)

        # Step 2: Recurrently Predict Future Latent States
        for t in range(1, seq_len):
            action_t = actions[:, t - 1]  # Get the action taken at time step t-1
            s_t = self.predictor(s_t, action_t)  # Predict the next latent state
            predicted_states.append(s_t.unsqueeze(1))  # Add the predicted latent state to the list

        # Step 3: Concatenate Predicted States Across Time
        predicted_states = torch.cat(predicted_states, dim=1)  # Concatenate along the time dimension

        return predicted_states
