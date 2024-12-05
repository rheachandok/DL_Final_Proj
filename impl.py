
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class EncoderNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(EncoderNetwork, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully Connected Layer
        self.fc = nn.Linear(128 * 9 * 9, hidden_dim)
        self.fc_bn = nn.BatchNorm1d(hidden_dim)
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc_bn(self.fc(x))
        x = F.relu(x)
        x = self.projector(x)
        return x


class PredictorNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PredictorNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        # Projection Head
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x

class JEPA(nn.Module):

    def __init__(self, input_channels, hidden_dim, action_dim):
        super(JEPA, self).__init__()
        self.encoder = EncoderNetwork(input_channels, hidden_dim)
        self.predictor = PredictorNetwork(hidden_dim, action_dim)
      
        self.repr_dim = hidden_dim

    def forward(self, states1, actions, states2=None, return_actual_embeddings=False):
        embeddings1_pred = self._process_sequence(states1, actions)
        if return_actual_embeddings:
            embeddings1_actual = self._encode_sequence(states1)
        if states2 is not None:
            embeddings2_pred = self._process_sequence(states2, actions)
            if return_actual_embeddings:
                embeddings2_actual = self._encode_sequence(states2)
            if return_actual_embeddings:
                return embeddings1_pred, embeddings2_pred, embeddings1_actual, embeddings2_actual
            else:
                return embeddings1_pred, embeddings2_pred
        else:
            if return_actual_embeddings:
                return embeddings1_pred, embeddings1_actual
            else:
                return embeddings1_pred

    def _encode_sequence(self, states):
        # Encode each state in the sequence
        embeddings = [self.encoder(state) for state in states.transpose(0, 1)]
        embeddings = torch.stack(embeddings, dim=1)  # Shape: [batch_size, seq_len, hidden_dim]
        return embeddings


    def _process_sequence(self, states, actions):
        batch_size, seq_len, channels, height, width = states.size()
        predicted_states = []

        # Encode the initial observation
        s_t = self.encoder(states[:, 0])
        predicted_states.append(s_t.unsqueeze(1))

        # Recurrently predict future latent states
        for t in range(1, seq_len):
            action_t = actions[:, t - 1]
            s_t = self.predictor(s_t, action_t)
            predicted_states.append(s_t.unsqueeze(1))

        # Concatenate predicted states across time
        predicted_states = torch.cat(predicted_states, dim=1)
        return predicted_states


def vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    Compute the VicReg loss between two batches of embeddings x and y.

    Args:
        x (torch.Tensor): Embeddings from view 1, shape (batch_size, feature_dim)
        y (torch.Tensor): Embeddings from view 2, shape (batch_size, feature_dim)
        sim_weight (float): Weight for the invariance (similarity) term
        var_weight (float): Weight for the variance term
        cov_weight (float): Weight for the covariance term

    Returns:
        torch.Tensor: The computed VicReg loss
    """
    # Invariance loss (Mean Squared Error)
    invariance_loss = F.mse_loss(x, y)

    # Variance loss
    def variance_loss(z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))

    var_loss = variance_loss(x) + variance_loss(y)

    # Covariance loss
    def covariance_loss(z):
        batch_size, feature_dim = z.size()
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (batch_size - 1)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        return (off_diagonal ** 2).sum() / feature_dim

    cov_loss = covariance_loss(x) + covariance_loss(y)

    # Total VicReg loss
    loss = sim_weight * invariance_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss


def train_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    """
    Trains the JEPA model using VicReg.
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0  # Accumulates the loss over the epoch

        with tqdm(total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit='batch') as pbar:
            for states1, states2, actions in dataloader:
                states1 = states1.to(device)
                states2 = states2.to(device)
                actions = actions.to(device)

                optimizer.zero_grad()

                # Forward pass with both augmented views
                embeddings1_pred, embeddings2_pred, embeddings1_actual, embeddings2_actual = model(
                    states1, actions, states2=states2, return_actual_embeddings=True
                )

                # Flatten embeddings
                batch_size, seq_len, hidden_dim = embeddings1_pred.size()
                embeddings1_pred_flat = embeddings1_pred.view(batch_size * seq_len, hidden_dim)
                embeddings2_pred_flat = embeddings2_pred.view(batch_size * seq_len, hidden_dim)
                embeddings1_actual_flat = embeddings1_actual.view(batch_size * seq_len, hidden_dim)
                embeddings2_actual_flat = embeddings2_actual.view(batch_size * seq_len, hidden_dim)

                # Compute VicReg loss between predicted embeddings
                vicreg_loss_value = vicreg_loss(embeddings1_pred_flat, embeddings2_pred_flat)

                # Compute prediction loss between predicted and actual embeddings
                loss_pred1 = F.mse_loss(embeddings1_pred_flat, embeddings1_actual_flat)
                loss_pred2 = F.mse_loss(embeddings2_pred_flat, embeddings2_actual_flat)
                prediction_loss = (loss_pred1 + loss_pred2) / 2

                # Total loss
                alpha = 1.0  # Hyperparameter to balance losses
                batch_loss = vicreg_loss_value + alpha * prediction_loss

                # Backward pass and optimization
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                # Accumulate total loss
                total_loss += batch_loss.item()
                
                # Update tqdm progress bar
                pbar.set_postfix({'Loss': f'{batch_loss.item():.6f}'})
                pbar.update(1)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
