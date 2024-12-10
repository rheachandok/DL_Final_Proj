import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training

# Optimized Encoder Network
class EncoderNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(EncoderNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 8 * 8, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.dropout(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

# Optimized Predictor Network
class PredictorNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PredictorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        x = self.fc(x)
        return x

# JEPA Model
class JEPA(nn.Module):
    def __init__(self, input_channels, hidden_dim, action_dim):
        super(JEPA, self).__init__()
        self.encoder = EncoderNetwork(input_channels, hidden_dim)
        self.predictor = PredictorNetwork(hidden_dim, action_dim)

    def forward(self, states, actions):
        batch_size, seq_len, channels, height, width = states.size()
        predicted_states = []

        s_t = self.encoder(states[:, 0])  # Encode initial state
        predicted_states.append(s_t.unsqueeze(1))

        for t in range(1, seq_len):
            action_t = actions[:, t - 1]
            s_t = self.predictor(s_t, action_t)
            predicted_states.append(s_t.unsqueeze(1))

        predicted_states = torch.cat(predicted_states, dim=1)
        return predicted_states

# Optimized VICReg Loss Function
def vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    invariance_loss = F.mse_loss(x, y)

    std_x = torch.sqrt(x.var(dim=0) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    batch_size, feature_dim = x.size()
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_loss = (cov_x - torch.diag(torch.diag(cov_x))).pow(2).sum()

    return sim_weight * invariance_loss + var_weight * var_loss + cov_weight * cov_loss

# Training Function
def train_model(model, dataloader, optimizer, scheduler, num_epochs, device):
    scaler = GradScaler()  # Mixed precision
    loss_history = []

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for states, actions, targets in dataloader:
                states, actions, targets = states.to(device), actions.to(device), targets.to(device)

                optimizer.zero_grad()
                with autocast():  # Enable mixed precision
                    predicted_states = model(states, actions)
                    loss = vicreg_loss(
                        predicted_states.view(-1, predicted_states.size(-1)),
                        targets.view(-1, targets.size(-1))
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()  # Adjust learning rate
                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    return loss_history

# Mock Data Generation
def get_mock_dataloader(batch_size, seq_len, channels, height, width, hidden_dim, action_dim):
    states = torch.randn(batch_size, seq_len, channels, height, width)
    actions = torch.randn(batch_size, seq_len - 1, action_dim)
    targets = torch.randn(batch_size, seq_len, hidden_dim)
    dataset = TensorDataset(states, actions, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Main Execution
if __name__ == "__main__":
    # Parameters
    batch_size = 8
    seq_len = 17
    channels, height, width = 3, 64, 64
    hidden_dim = 128
    action_dim = 10
    num_epochs = 20
    learning_rate = 0.001

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPA(input_channels=channels, hidden_dim=hidden_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

    dataloader = get_mock_dataloader(batch_size, seq_len, channels, height, width, hidden_dim, action_dim)
    loss_history = train_model(model, dataloader, optimizer, scheduler, num_epochs, device)

    # Plot Loss
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
