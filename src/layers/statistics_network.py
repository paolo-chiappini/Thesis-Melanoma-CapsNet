import torch
import torch.nn as nn


class StatisticsNetwork(nn.Module):
    """
    Statistics Network for Mutual Information estimation based on MINE (https://arxiv.org/pdf/1801.04062)
    """

    def __init__(self, pose_dim: int, hidden_dim: int = 128):
        super(StatisticsNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(pose_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        """
        Args:
            z (torch.Tensor): capsule pose (N, pose_dim)
            y (torch.Tensor): binary label (N, 1)
        """
        input_vec = torch.cat([z, y], dim=1)
        return self.network(input_vec)
