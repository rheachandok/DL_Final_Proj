import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        """
        Encodes state images into a latent embedding space.

        Args:
            embedding_dim (int): Dimension of the output embedding.
        """
        super(StateEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # Input channels: 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 8, 8)
        )

        # Adaptive Pooling to retain spatial detail (e.g., 4x4)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output: (128, 4, 4)

        # Fully connected layer to project to embedding space
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)

    def forward(self, x):
        """
        Forward pass for the StateEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 2, 65, 65].

        Returns:
            torch.Tensor: Embedding tensor of shape [B, embedding_dim].
        """
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [B, 128*4*4]
        x = self.fc(x)
        return x  # [B, embedding_dim]


class ActionEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        """
        Encodes action vectors into a latent embedding space.

        Args:
            embedding_dim (int): Dimension of the output embedding. Default is 256.
        """
        super(ActionEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),  # Input: (delta_x, delta_y)
            nn.ReLU(),
            nn.Linear(128, embedding_dim)  # Project to 256-dimensional space
        )

    def forward(self, x):
        """
        Forward pass for the ActionEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T-1, 2].

        Returns:
            torch.Tensor: Embedding tensor of shape [B, T-1, embedding_dim].
        """
        B, T, A = x.size()
        x = x.view(-1, A)  # [B*(T-1), 2]
        x = self.fc(x)      # [B*(T-1), 256]
        x = x.view(B, T, -1)  # [B, T-1, 256]
        return x  # [B, T-1, 256]


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, combined_embeddings):
        # combined_embeddings: [B, T, input_dim] (State + Action concatenated over time)
        output, _ = self.gru(combined_embeddings)  # [B, T, hidden_dim]
        output = self.fc(output)  # Map to state_latent_dim
        return output  # [B, T, output_dim]


class JEPA(nn.Module):
    def __init__(self, state_latent_dim=256, action_latent_dim=256, hidden_dim=128):
        """
        Joint Embedding Predictive Architecture (JEPA) model.

        Args:
            state_latent_dim (int): Dimension of the state embeddings. Default is 256.
            action_latent_dim (int): Dimension of the action embeddings. Default is 256.
            hidden_dim (int): Dimension of the Predictor's hidden layer. Default is 128.
        """
        super(JEPA, self).__init__()
        self.state_encoder = StateEncoder(embedding_dim=state_latent_dim)
        self.action_encoder = ActionEncoder(embedding_dim=action_latent_dim)

        # Predictor maps from concatenated state and action embeddings to next state embedding
        self.predictor = Predictor(
            input_dim=state_latent_dim + action_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=state_latent_dim
        )

        self.repr_dim = state_latent_dim


    def forward(self, states, actions):
        """
        Predict a sequence of future state embeddings starting from an initial state.

        Args:
            states (torch.Tensor): Input tensor of shape [B, T, C, H, W] or [B, C, H, W].
            actions (torch.Tensor): Sequence of actions of shape [B, T, 2].

        Returns:
            torch.Tensor: Sequence of predicted state embeddings of shape [B, T+1, state_latent_dim].
        """
        B, T, _ = actions.size()  # B: Batch size, T: Number of timesteps

        # Extract and encode the initial state
        if states.ndim == 5:
            # Time-series input: [B, T, C, H, W]
            initial_state = states[:, 0, :, :, :]  # Extract the first state
        elif states.ndim == 4:
            # Single state input: [B, C, H, W]
            initial_state = states
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}. Expected 4D or 5D tensor.")

        initial_state_embedding = self.state_encoder(initial_state)  # [B, state_latent_dim]

        # Encode actions
        action_embeddings = self.action_encoder(actions)  # [B, T, action_latent_dim]

        # Combine state and action embeddings
        initial_state_embedding = initial_state_embedding.unsqueeze(1)  # [B, 1, state_latent_dim]
        repeated_initial_state = initial_state_embedding.repeat(1, T, 1)  # [B, T, state_latent_dim]
        combined_embeddings = torch.cat((repeated_initial_state, action_embeddings), dim=2)  # [B, T, input_dim]

        # Predict future states
        predicted_embeddings = self.predictor(combined_embeddings)  # [B, T, state_latent_dim]

        # Include the initial state embedding in the sequence
        predicted_embeddings = torch.cat(
            (initial_state_embedding, predicted_embeddings), dim=1
        )  # [B, T+1, state_latent_dim]

        return predicted_embeddings
