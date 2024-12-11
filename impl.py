import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(StateEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(ActionEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        B, T, A = x.size()
        x = x.view(-1, A)  # Flatten across batch and sequence
        x = self.fc(x)
        x = x.view(B, T, -1)  # Reshape back to [B, T, embedding_dim]
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, dropout=0.2, num_layers=2):
        """
        Temporal Predictor using GRU for sequence modeling.

        Args:
            input_dim (int): Dimension of the input feature vector (state + action).
            hidden_dim (int): Dimension of the GRU's hidden state.
            output_dim (int): Dimension of the output feature vector.
            dropout (float): Dropout rate for regularization.
            num_layers (int): Number of GRU layers.
        """
        super(Predictor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # Dropout only applies for num_layers > 1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, states, actions):
        """
        Forward pass for the GRU-based Predictor.

        Args:
            states (torch.Tensor): State embeddings of shape [B, T, state_dim].
            actions (torch.Tensor): Action embeddings of shape [B, T, action_dim].

        Returns:
            torch.Tensor: Predicted state embeddings of shape [B, T, output_dim].
        """
        # Concatenate states and actions along the feature dimension
        x = torch.cat((states, actions), dim=2)  # [B, T, state_dim + action_dim]

        # Pass through GRU
        gru_out, _ = self.gru(x)  # GRU output: [B, T, hidden_dim]

        # Map GRU outputs to desired output_dim
        output = self.fc(gru_out)  # [B, T, output_dim]

        return output



class JEPA(nn.Module):
    def __init__(self, state_latent_dim=256, action_latent_dim=256, hidden_dim=128):
        super(JEPA, self).__init__()
        self.state_encoder = StateEncoder(embedding_dim=state_latent_dim)
        self.action_encoder = ActionEncoder(embedding_dim=action_latent_dim)
        self.predictor = Predictor(
            input_dim=state_latent_dim + action_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=state_latent_dim
        )
        self.repr_dim = state_latent_dim

    def forward(self, initial_state, actions):
        """
        Args:
            initial_state (torch.Tensor): Initial state image of shape [B, 2, 65, 65].
            actions (torch.Tensor): Sequence of actions of shape [B, T, 2].

        Returns:
            torch.Tensor: Sequence of predicted states of shape [B, T+1, 256].
        """
        B, T, _ = actions.size()

        # Encode the initial state
        current_state_embedding = self.state_encoder(initial_state)  # [B, 256]
        current_state_embedding = current_state_embedding.unsqueeze(1)  # [B, 1, 256]

        # Encode all actions
        action_embeddings = self.action_encoder(actions)  # [B, T, 256]

        # Expand initial state to match action sequence length
        state_sequence = current_state_embedding.repeat(1, T, 1)  # [B, T, 256]

        # Predict the next states
        predicted_states = self.predictor(state_sequence, action_embeddings)  # [B, T, 256]

        # Concatenate initial state with predicted states
        state_sequence = torch.cat((current_state_embedding, predicted_states), dim=1)  # [B, T+1, 256]

        return state_sequence
