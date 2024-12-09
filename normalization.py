import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

class Normalizer:
    def __init__(self, mean=None, std=None, device='cuda'):
        self.mean = mean
        self.std = std
        self.device = device

    def compute_embedding_stats(self, model, data_loader, device):
        """
        Computes mean and std from the training data.
        """
        all_embeddings = []
        for batch in tqdm(data_loader, desc="Computing Normalization Stats"):
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            # Encode target embeddings externally
            target_states = states[:, 16, :, :, :]  # Assuming t=16 is the target
            with torch.no_grad():
                Sy = model.state_encoder(target_states)  # [B, D]
            all_embeddings.append(Sy)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.mean = all_embeddings.mean(dim=0)
        self.std = all_embeddings.std(dim=0) + 1e-8  # Avoid division by zero

    def normalize_embeddings(self, embeddings):
        """
        Normalizes embeddings using the computed mean and std.
        """
        return (embeddings - self.mean) / self.std

    def denormalize(self, embeddings):
        """
        Denormalizes embeddings back to original scale.
        """
        return embeddings * self.std + self.mean
