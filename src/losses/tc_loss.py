import numpy as np
import torch
import torch.nn as nn


class TotalCorrelationLoss(nn.Module):
    def __init__(self, loss_beta=1.0):
        super(TotalCorrelationLoss, self).__init__()
        self.loss_beta = loss_beta

    def forward(self, z_activations):
        # flatten activations from (N, num_attr, pose_dim) to (N, num_attr * pose_dim)
        N, num_attr, pose_dim = z_activations.shape
        z_activations = z_activations.reshape(N, -1)

        N, K = z_activations.shape

        # normalize poses
        poses_mean = z_activations.mean(dim=0, keepdim=True)
        poses_std = z_activations.std(dim=0, keepdim=True)
        z_activations = (z_activations - poses_mean) / (poses_std + 1e-6)

        pairwise_log_prob_joint = compute_pairwise_log_gaussian(
            z_activations, z_activations
        )

        log_q_z_given_x = torch.logsumexp(pairwise_log_prob_joint, dim=1) - np.log(N)

        log_q_z_j = torch.zeros(N, K)
        for j in range(K):
            capsule_j_activations = z_activations[:, j].unsqueeze(1)  # shape (N, 1)
            pairwise_log_prob_marginal = compute_pairwise_log_gaussian(
                capsule_j_activations, capsule_j_activations
            )
            log_q_z_j[:, j] = torch.logsumexp(
                pairwise_log_prob_marginal, dim=1
            ) - np.log(N)

        sum_log_q_z_j = torch.sum(log_q_z_j, dim=1)

        device = z_activations.device
        log_q_z_given_x = log_q_z_given_x.to(device)
        sum_log_q_z_j = sum_log_q_z_j.to(device)

        tc_loss = torch.mean(log_q_z_given_x - sum_log_q_z_j)

        return tc_loss * self.loss_beta


def compute_pairwise_log_gaussian(x, mu, sigma=1.0):
    """
    Computes the pairwise log-likelihood of a Gaussian distribution.

    This function calculates log p(x_i | mu_j) for all i, j, assuming a
    multivariate Gaussian with a diagonal covariance matrix sigma^2 * I.

    Args:
        x (torch.Tensor): A tensor of shape (N, K), representing N data points
                          of K dimensions. In our case, the batch of capsule activations.
        mu (torch.Tensor): A tensor of shape (M, K), representing M centers
                           of Gaussian distributions. Here, it's the same as x, so M=N.
        sigma (float): The standard deviation of the Gaussian distributions. This is
                       a hyperparameter.

    Returns:
        torch.Tensor: A tensor of shape (N, M) where the element (i, j) is
                      log p(x_i | mu_j).
    """
    # Ensure inputs are 2D tensors
    if x.dim() != 2 or mu.dim() != 2:
        raise ValueError("Inputs must be 2D tensors.")

    N, K = x.shape
    M, _ = mu.shape

    # Use broadcasting to efficiently compute the pairwise squared distances.
    # x_expanded: (N, 1, K)
    # mu_expanded: (1, M, K)
    # After subtraction, result is (N, M, K), where result[i, j, :] = x[i, :] - mu[j, :]
    x_expanded = x.unsqueeze(1)
    mu_expanded = mu.unsqueeze(0)

    # Calculate the squared difference and sum over the K dimensions.
    # squared_dist is the squared Euclidean distance ||x_i - mu_j||^2
    # Shape: (N, M)
    squared_dist = torch.sum((x_expanded - mu_expanded) ** 2, dim=2)

    # The constant term is calculated once.
    log_prob_const = -0.5 * K * np.log(2 * np.pi) - K * np.log(sigma)

    log_prob = log_prob_const - (squared_dist / (2 * sigma**2))

    return log_prob
    return log_prob
