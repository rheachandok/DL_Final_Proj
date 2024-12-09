import torch.nn.functional as F
import torch.nn as nn
import torch

def invariance_loss(Sy_hat, Sy):
    return F.mse_loss(Sy_hat, Sy)

def variance_loss(embeddings, eps=1e-4, threshold=1.0):
    std = embeddings.std(dim=0)
    variance = torch.mean(F.relu(threshold - std))
    return variance

def covariance_loss(embeddings):
    B, D = embeddings.size()
    embeddings = embeddings - embeddings.mean(dim=0)
    cov = (embeddings.T @ embeddings) / (B - 1)
    cov = cov - torch.diag(torch.diag(cov))
    loss = torch.sum(cov ** 2) / D
    return loss

class VICRegLoss(nn.Module):
    def __init__(self, lambda_invariance=25.0, lambda_variance=25.0, lambda_covariance=1.0):
        super(VICRegLoss, self).__init__()
        self.lambda_invariance = lambda_invariance
        self.lambda_variance = lambda_variance
        self.lambda_covariance = lambda_covariance

    def forward(self, Sy_hat, Sy):
        inv_loss = invariance_loss(Sy_hat, Sy)
        var_loss = variance_loss(Sy_hat) + variance_loss(Sy)
        cov_loss = covariance_loss(Sy_hat) + covariance_loss(Sy)
        total_loss = (
            self.lambda_invariance * inv_loss +
            self.lambda_variance * var_loss +
            self.lambda_covariance * cov_loss
        )
        return total_loss, inv_loss, var_loss, cov_loss
