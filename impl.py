import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=256):
        """
        Encoder: Processes the observation images (64x64x2) into a latent representation.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(256 * 8 * 8, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=2):
        """
        Predictor: Predicts the next latent state given the current state and action.
        """
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class JEPA(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=256, action_dim=2):
        """
        Joint Embedding Prediction Architecture (JEPA) model.
        """
        super(JEPA, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, hidden_dim=hidden_dim)
        self.predictor = Predictor(hidden_dim=hidden_dim, action_dim=action_dim)
        self.repr_dim = hidden_dim

    def forward(self, states, actions):
        """
        Forward pass for the JEPA model.
        Args:
            states: (batch_size, sequence_length, 2, 64, 64)
            actions: (batch_size, sequence_length-1, 2)
        Returns:
            pred_states: (sequence_length-1, batch_size, hidden_dim)
        """
        batch_size, seq_len, _, _, _ = states.size()

        # Encode all states into latent representations
        encoded_states = torch.stack([self.encoder(states[:, t]) for t in range(seq_len)], dim=1)

        # Predict the next latent states
        pred_states = []
        for t in range(seq_len - 1):
            pred_state = self.predictor(encoded_states[:, t], actions[:, t])
            pred_states.append(pred_state)

        return torch.stack(pred_states, dim=0)  # Shape: (seq_len-1, batch_size, hidden_dim)
