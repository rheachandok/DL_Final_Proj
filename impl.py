import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces 65x65 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces 32x32 -> 16x16
        )
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 65, 65)  # Adjust to match your input
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = int(torch.prod(torch.tensor(conv_output.shape[1:])))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        print(f"Shape after conv_layers: {x.shape}")  # Debugging shape
        x = self.fc_layers(x)
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=2):
        """
        Predictor: Predicts the next latent state given the current state and action.
        """
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state, action):
        return self.predictor(torch.cat([state, action], dim=-1))


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
        print("Shapes:",states.shape)
        pred_states = []

        # Encode all states into latent representations
        encoded_states = torch.stack([self.encoder(states[:, t]) for t in range(seq_len)], dim=1)
        print(f"Encoded states shape: {encoded_states.shape}")  # Debugging line

        for t in range(seq_len - 1):
            print(f"Encoding at time {t}: {encoded_states[:, t].shape}")
            print(f"Action at time {t}: {actions[:, t].shape}")
            pred_state = self.predictor(encoded_states[:, t], actions[:, t])
            pred_states.append(pred_state)


        return torch.stack(pred_states, dim=0)

