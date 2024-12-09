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
    def __init__(self, input_dim, hidden_dim=256):
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
        
        # Input to temporal model is state_embedding + action_embedding
        self.temporal_input_dim = state_latent_dim + action_latent_dim
        self.temporal_model = TemporalModel(input_dim=self.temporal_input_dim, hidden_dim=hidden_dim)
        
        # Predictor maps from hidden_dim (Sx) to state_latent_dim (Sy)
        self.predictor = Predictor(input_dim=hidden_dim, output_dim=state_latent_dim)


    def forward(self, states, actions):
            # states: [B, 17, 2, 65, 65]
            # actions: [B, 16, 2]
            
            B = states.shape[0]
            
            # Encode all states
            states_reshaped = states.view(B*17, 2, 65, 65)
            state_embeds = self.state_encoder(states_reshaped) # [B*17, state_latent_dim]
            state_embeds = state_embeds.view(B, 17, -1) # [B, 17, state_latent_dim]
            
            # Take initial states (t=0,...,15) and final state (t=16)
            state_embeds_init = state_embeds[:, 0:16, :] # [B, 16, state_latent_dim]
            state_embed_final = state_embeds[:, 16, :]   # Sy
            
            # Encode actions
            action_embeds = self.action_encoder(actions) # [B, 16, action_latent_dim]
            
            # Concatenate state and action embeddings for temporal model
            seq = torch.cat([state_embeds_init, action_embeds], dim=-1) # [B,16, state_latent_dim+action_latent_dim]
            
            # Get Sx from temporal model
            Sx = self.temporal_model(seq) # [B, hidden_dim]
            
            # Predict Sy_hat
            Sy_hat = self.predictor(Sx) # [B, state_latent_dim]
            
            return Sy_hat
