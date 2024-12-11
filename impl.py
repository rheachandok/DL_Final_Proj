import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, dropout=0.2, num_layers=2):
        super(Predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM layers for temporal prediction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layers for mapping to the output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, states, actions):
        """
        Forward pass for the TemporalPredictor.

        Args:
            states: Tensor of shape [B, T, state_dim].
            actions: Tensor of shape [B, T, action_dim].

        Returns:
            Tensor of shape [B, T, output_dim].
        """
        # Concatenate states and actions along the feature dimension
        x = torch.cat((states, actions), dim=2)  # [B, T, state_dim + action_dim]

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, hidden_dim]

        # Map LSTM outputs to desired output_dim
        output = self.fc(lstm_out)  # [B, T, output_dim]

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

    def forward(self, initial_state, actions):
        """
        Forward pass for generating a sequence of states.

        Args:
            initial_state (torch.Tensor): The initial state image of shape [B, 2, 65, 65].
            actions (torch.Tensor): The sequence of actions of shape [B, 16, 2].

        Returns:
            torch.Tensor: The sequence of states of shape [B, 17, 256].
        """
        B, T, _ = actions.size()  # B: batch size, T: sequence length (16 actions)

        # Encode the initial state
        current_state_embedding = self.state_encoder(initial_state)  # [B, 256]

        # Store the sequence of state embeddings
        state_embeddings = [current_state_embedding]

        # Encode actions
        action_embeddings = self.action_encoder(actions)  # [B, 16, 256]

        # Iteratively predict the next state
        for t in range(T):  # Loop over the 16 actions
            current_action_embedding = action_embeddings[:, t, :]  # [B, 256]
            next_state_embedding = self.predictor(current_state_embedding, current_action_embedding)
            state_embeddings.append(next_state_embedding)  # Append to the sequence
            current_state_embedding = next_state_embedding  # Update the current state

        # Concatenate all state embeddings along the time axis
        state_embeddings = torch.stack(state_embeddings, dim=1)  # [B, 17, 256]

        return state_embeddings  # Return all 17 state embeddings