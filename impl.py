import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(StateEncoder, self).__init__()
        # A simple CNN: adjust layers as necessary
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,33,33]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [B,128,17,17]
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# [B,128,9,9]
            nn.ReLU(),
        )
        # Flatten and linear layer to get latent_dim
        self.fc = nn.Linear(128*9*9, latent_dim)

    def forward(self, x):
        # x: [B, 2, 65, 65]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h # [B, latent_dim]


class ActionEncoder(nn.Module):
    def __init__(self, action_dim=2, latent_dim=32):
        super(ActionEncoder, self).__init__()
        self.fc = nn.Linear(action_dim, latent_dim)
        
    def forward(self, actions):
        # actions: [B, T, 2]
        B, T, _ = actions.shape
        actions_flat = actions.view(B*T, -1)
        h = self.fc(actions_flat)
        h = h.view(B, T, -1)
        return h # [B, T, latent_dim]


class TemporalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(TemporalModel, self).__init__()
        # An LSTM to aggregate sequence information
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, seq):
        # seq: [B, T, input_dim]
        _, (h_n, _) = self.lstm(seq)
        # h_n: [1, B, hidden_dim] -> [B, hidden_dim]
        return h_n.squeeze(0)

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim)  # Normalization layer
        
    def forward(self, x):
        x = self.fc(x)               # [B, output_dim]
        x = self.layer_norm(x)       # Normalize encodings
        return x
    

class JEPA(nn.Module):
    def __init__(self, state_latent_dim=128, action_latent_dim=32, hidden_dim=256, train=False):
        super(JEPA, self).__init__()
        self.state_encoder = StateEncoder(latent_dim=state_latent_dim)
        self.action_encoder = ActionEncoder(latent_dim=action_latent_dim)
        
        # Temporal model input: state_embedding + action_embedding
        self.temporal_input_dim = state_latent_dim + action_latent_dim
        self.temporal_model = TemporalModel(input_dim=self.temporal_input_dim, hidden_dim=hidden_dim)
        
        # Predictor maps from hidden_dim (Sx) to state_latent_dim (Sy)
        self.predictor = Predictor(input_dim=hidden_dim, output_dim=state_latent_dim)

    def forward(self, states, actions, trajectory_length=17):
        B = states.shape[0]

        # List to store the predicted states
        predicted_states = []

        # Encode the initial state (t=0)
        initial_state = states[:, 0, :, :, :]  # Shape [B, 2, 65, 65]
        state_embed_initial = self.state_encoder(initial_state)  # [B, state_latent_dim]
        
        # Encode actions (for the first time step)
        action_embed = self.action_encoder(actions)  # Shape [B, 16, action_latent_dim]

        # Iterate through the trajectory to predict the next state
        state_embed = state_embed_initial  # Initialize with the first state embedding

        for t in range(trajectory_length - 1):  # Predict next states iteratively
            # Concatenate the current state embedding with action embeddings for the current time step
            seq_input = torch.cat([state_embed.unsqueeze(1), action_embed[:, t, :].unsqueeze(1)], dim=-1)  # [B, 1, state_latent_dim + action_latent_dim]

            # Get the hidden representation (Sx) from the temporal model
            Sx = self.temporal_model(seq_input)  # Shape [B, hidden_dim]
            
            # Predict the next state (Sy_hat)
            Sy_hat = self.predictor(Sx)  # [B, state_latent_dim]
            predicted_states.append(Sy_hat)

            # Update state_embed for the next iteration (prediction for t+1)
            state_embed = Sy_hat

        return torch.stack(predicted_states, dim=1)  # [B, T-1, state_latent_dim], sequence of predicted states
