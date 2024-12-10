import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(StateEncoder, self).__init__()
        # Updated CNN with BatchNorm and increased depth
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h


class ActionEncoder(nn.Module):
    def __init__(self, action_dim=2, latent_dim=32):
        super(ActionEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, actions):
        B, T, _ = actions.shape
        actions_flat = actions.view(B * T, -1)
        h = self.fc(actions_flat)
        h = h.view(B, T, -1)
        return h


class TemporalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2):
        super(TemporalModel, self).__init__()
        # Replace LSTM with GRU for simplicity and efficiency
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, seq):
        _, h_n = self.gru(seq)
        return h_n[-1]  # Use the last hidden state


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=0.3, training=self.training)  # Dropout regularization
        h = self.fc2(h)
        return self.layer_norm(h)


class JEPA(nn.Module):
    def __init__(self, state_latent_dim=256, action_latent_dim=32, hidden_dim=256):
        super(JEPA, self).__init__()
        self.state_encoder = StateEncoder(latent_dim=state_latent_dim)
        self.action_encoder = ActionEncoder(latent_dim=action_latent_dim)
        self.temporal_model = TemporalModel(input_dim=state_latent_dim + action_latent_dim, hidden_dim=hidden_dim)
        self.predictor = Predictor(input_dim=hidden_dim, output_dim=state_latent_dim)
        self.repr_dim = hidden_dim

    def forward(self, states, actions, trajectory_length=17, teacher_forcing_ratio=0.5):
        B = states.shape[0]
        predicted_states = []
        state_embed = self.state_encoder(states[:, 0])  # Initial state encoding

        action_embed = self.action_encoder(actions)  # Encode all actions
        predicted_states.append(state_embed)

        for t in range(trajectory_length - 1):
            seq_input = torch.cat([state_embed.unsqueeze(1), action_embed[:, t, :].unsqueeze(1)], dim=-1)
            Sx = self.temporal_model(seq_input)
            Sy_hat = self.predictor(Sx)

            # Teacher forcing
#            if torch.rand(1).item() < teacher_forcing_ratio:
 #               state_embed = self.state_encoder(states[:, t + 1])  # Ground truth state
  #          else:
            state_embed = Sy_hat

            predicted_states.append(Sy_hat)

        return torch.stack(predicted_states, dim=1)
