import torch.nn.functional as F
import torch.nn as nn
import torch

def invariance_loss(Sy_hat, Sy):
    """
    Measures the similarity between predicted and target embeddings.
    """
    return F.mse_loss(Sy_hat, Sy)

def slim_variance_loss(embeddings, eps=1e-4, min_threshold=1.0):
    """
    Encourages sufficient spread in embeddings by penalizing low variance.
    Uses a stronger constraint to prevent collapse.
    """
    std = embeddings.std(dim=0) + eps  # Avoid division by zero
    variance_loss = torch.mean(torch.clamp(min_threshold - std, min=0))  # Penalize low std
    return variance_loss

def slim_covariance_loss(embeddings, reg_weight=0.1):
    """
    Encourages decorrelation of embedding dimensions while reducing off-diagonal covariance.
    """
    B, D = embeddings.size()
    embeddings = embeddings - embeddings.mean(dim=0)  # Zero-center embeddings
    cov = (embeddings.T @ embeddings) / (B - 1)  # Covariance matrix
    cov_loss = (cov ** 2).sum() - torch.diagonal(cov).pow(2).sum()  # Off-diagonal terms
    return reg_weight * cov_loss / D  # Normalize by embedding dimension

class SLIMCRLoss(nn.Module):
    def __init__(self, lambda_invariance=1.0, lambda_variance=15.0, lambda_covariance=1.0):
        """
        SLIM-CR Loss: Combines invariance, variance, and covariance regularization.

        Args:
            lambda_invariance (float): Weight for invariance loss.
            lambda_variance (float): Weight for variance regularization.
            lambda_covariance (float): Weight for covariance regularization.
        """
        super(SLIMCRLoss, self).__init__()
        self.lambda_invariance = lambda_invariance
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance

    def forward(self, Sy_hat, Sy):
        """
        Computes the SLIM-CR loss.

        Args:
            Sy_hat (torch.Tensor): Predicted embeddings (B x D).
            Sy (torch.Tensor): Target embeddings (B x D).

        Returns:
            total_loss (torch.Tensor): Total SLIM-CR loss.
            inv_loss (torch.Tensor): Invariance loss.
            var_loss (torch.Tensor): Variance regularization loss.
            cov_loss (torch.Tensor): Covariance regularization loss.
        """
        # Invariance Loss: Align predictions with targets
        inv_loss = invariance_loss(Sy_hat, Sy)

        # Variance Loss: Ensure embeddings have sufficient variance
        var_loss = slim_variance_loss(Sy_hat) + slim_variance_loss(Sy)

        # Covariance Loss: Decorrelate embedding dimensions
        cov_loss = slim_covariance_loss(Sy_hat) + slim_covariance_loss(Sy)

        # Total Loss
        total_loss = (
            self.lambda_invariance * inv_loss +
            self.lambda_variance * var_loss +
            self.lambda_covariance * cov_loss
        )
        return total_loss, inv_loss, var_loss, cov_loss
