import torch.nn.functional as F
import torch.nn as nn
import torch

def invariance_loss(Sy_hat, Sy):
    """
    Measures the similarity between predicted and target embeddings.
    """
    return F.mse_loss(Sy_hat, Sy)

def variance_loss(embeddings, eps=1e-4, threshold=1.0):
    """
    Encourages sufficient spread in embeddings by penalizing low variance.
    """
    std = embeddings.std(dim=0) + eps  # Avoid division by zero
    variance = torch.mean(torch.clamp(threshold - std, min=0))  # Penalize low std
    return variance

def covariance_loss(embeddings):
    """
    Penalizes off-diagonal covariance terms to decorrelate embedding dimensions.
    """
    B, D = embeddings.size()
    embeddings = embeddings - embeddings.mean(dim=0)  # Zero-center embeddings
    cov = (embeddings.T @ embeddings) / (B - 1)  # Covariance matrix
    cov = cov - torch.diag(torch.diag(cov))  # Remove diagonal
    loss = torch.sum(cov ** 2) / D  # Penalize off-diagonal terms
    return loss

class VICRegLoss(nn.Module):
    def __init__(self, lambda_invariance=1.0, lambda_variance=25.0, lambda_covariance=1.0):
        """
        VICReg Loss: Combines invariance, variance, and covariance regularization.

        Args:
            lambda_invariance (float): Weight for invariance loss.
            lambda_variance (float): Weight for variance regularization.
            lambda_covariance (float): Weight for covariance regularization.
        """
        super(VICRegLoss, self).__init__()
        self.lambda_invariance = lambda_invariance
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance

    def forward(self, Sy_hat, Sy):
        """
        Computes the VICReg loss.

        Args:
            Sy_hat (torch.Tensor): Predicted embeddings (B x D).
            Sy (torch.Tensor): Target embeddings (B x D).

        Returns:
            total_loss (torch.Tensor): Total VICReg loss.
            inv_loss (torch.Tensor): Invariance loss.
            var_loss (torch.Tensor): Variance regularization loss.
            cov_loss (torch.Tensor): Covariance regularization loss.
        """
        # Invariance Loss: Align predictions with targets
        inv_loss = invariance_loss(Sy_hat, Sy)

        # Variance Loss: Ensure sufficient spread in embeddings
        var_loss = variance_loss(Sy_hat) + variance_loss(Sy)

        # Covariance Loss: Decorrelate embedding dimensions
        cov_loss = covariance_loss(Sy_hat) + covariance_loss(Sy)

        # Total Loss
        total_loss = (
            self.lambda_invariance * inv_loss +
            self.lambda_variance * var_loss +
            self.lambda_covariance * cov_loss
        )
        return total_loss, inv_loss, var_loss, cov_loss
