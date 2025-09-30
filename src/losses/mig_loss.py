# MIG Loss based on MINE (https://arxiv.org/pdf/1801.04062)

import torch
import torch.nn as nn

from layers.statistics_network import StatisticsNetwork


class MIGLoss(nn.Module):
    def __init__(
        self,
        num_attributes: int,
        pose_dim: int,
        lambda_align: float = 1.0,
        lambda_disentangle: float = 0.1,
        device: torch.device = "cuda",
        **kwargs
    ):
        super(MIGLoss, self).__init__()
        self.num_attributes = num_attributes
        self.pose_dim = pose_dim
        self.lambda_align = lambda_align
        self.lambda_disentangle = lambda_disentangle

        self.device = device

        self.alignment_estimators = nn.ModuleList(
            [
                StatisticsNetwork(pose_dim=pose_dim).to(device=device)
                for _ in range(num_attributes)
            ]
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_auxilliary(
        self, attribute_poses: torch.Tensor, va_labels: torch.Tensor
    ) -> torch.Tensor:
        # train MINE statistics network (see Algorithm 1 https://arxiv.org/pdf/1801.04062)
        detached_poses = attribute_poses.detach()

        mine_objective = 0.0
        for k in range(self.num_attributes):
            z_k = detached_poses[:, k, :]  # poses for attribute k
            y_k = va_labels[:, k].unsqueeze(1)  # labels for attribute k

            t_joint = self.alignment_estimators[k](z_k, y_k)

            y_k_shuffled = y_k[
                torch.randperm(y_k.size(0))
            ]  # shuffle to break correlation
            t_marginal = self.alignment_estimators[k](z_k, y_k_shuffled)

            mine_objective_k = torch.mean(t_joint) - torch.log(
                torch.mean(torch.exp(t_marginal))
            )
            mine_objective -= mine_objective_k

        # update statistics networks
        self.optimizer.zero_grad()
        mine_objective.backward()
        self.optimizer.step()

    def forward(
        self, attribute_poses: torch.Tensor, va_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            attribute_poses (torch.Tensor): attribute capsules poses (N, K, pose_dim)
            va_labels (torch.Tensor): attribute gt labels (N, K)
        """
        loss_align = 0.0
        loss_disentangle = 0.0
        for k in range(self.num_attributes):
            z_k = attribute_poses[:, k, :]
            y_k = va_labels[:, k].unsqueeze(1)

            y_k_shuffled = y_k[torch.randperm(y_k.size(0))]
            mi_estimate_k = torch.mean(
                self.alignment_estimators[k](z_k, y_k)
            ) - torch.log(
                torch.mean(torch.exp(self.alignment_estimators[k](z_k, y_k_shuffled)))
            )

            # maximize alignment along diagonal
            loss_align -= mi_estimate_k

            # check if in align-only mode
            if self.lambda_disentangle != 0:
                for j in range(self.num_attributes):
                    if k == j:
                        continue

                    y_j = va_labels[:, j].unsqueeze(1)

                    y_j_shuffled = y_j[torch.randperm(y_j.size(0))]
                    mi_estimate_kj = torch.mean(
                        self.alignment_estimators[k](z_k, y_j)
                    ) - torch.log(
                        torch.mean(
                            torch.exp(self.alignment_estimators[k](z_k, y_j_shuffled))
                        )
                    )

                    # minimize MI on off-diagonal terms
                    loss_disentangle += mi_estimate_kj

        total_mig_loss = (
            self.lambda_align * loss_align + self.lambda_disentangle * loss_disentangle
        )

        return total_mig_loss
