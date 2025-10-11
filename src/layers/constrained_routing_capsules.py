from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedRoutingCapsules(nn.Module):

    def __init__(
        self,
        num_primary_caps: int,
        num_attribute_caps: int,
        primary_dim: int,
        pose_dim: int,
        num_iterations: int = 3,
    ):
        super(ConstrainedRoutingCapsules, self).__init__()

        self.num_iterations = num_iterations
        self.num_primary = num_primary_caps
        self.num_attributes = num_attribute_caps
        self.primary_dim = primary_dim
        self.pose_dim = pose_dim

        self.routing_constraint_W = nn.Parameter(
            torch.randn(num_primary_caps, num_attribute_caps) * 0.1
        )

        self.W_t = nn.Parameter(
            torch.randn(
                self.num_primary, self.num_attributes, self.pose_dim, self.primary_dim
            )
        )
        nn.init.kaiming_uniform_(self.W_t, a=0.01)

    def forward(
        self, primary_caps_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constrained dynamic routing.

        Args:
            primary_caps_output (torch.Tensor): ourput of primary caps layer (u_i). (B, num_primary, dim_primary)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple contianing
                1. the pose vectors of the attribute capsule (v_j). (B, num_attribute, dim_attribute)
                2. the coupling coefficients between primary and attribute capsules (c_ij). (B, num_primary, num_attribute)
        """
        B, num_primary, dim_primary = primary_caps_output.shape
        u_i = primary_caps_output.unsqueeze(2).unsqueeze(
            -1
        )  # (B, num_primary, 1, dim_primary, 1)

        # W_t @ u_i = (num_primary, num_attribute, dim_attribute, dim_primary) @ (B, num_primary, num_attribute, dim_attribute)
        u_hat = torch.matmul(self.W_t, u_i).squeeze(-1)

        b_ij = torch.zeros(
            (B, self.num_primary, self.num_attributes), device=u_hat.device
        )

        for r in range(self.num_iterations):
            # Add the learned constraint W to the routing logits
            constrained_b_ij = b_ij + self.routing_constraint_W.unsqueeze(0)
            # constrained_b_ij = b_ij

            c_ij = F.softmax(constrained_b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(
                dim=1
            )  # (B, num_attribute, dim_attribute)

            v_j = self._squash(s_j)

            if r < self.num_iterations - 1:
                agreement = torch.sum(
                    v_j.unsqueeze(1) * u_hat, dim=-1
                )  # (B, num_primary, num_attribute)
                b_ij = b_ij + agreement.detach()

        return s_j, c_ij

    def _squash(self, input_tensor: torch.Tensor) -> torch.Tensor:
        squared_norm = (input_tensor**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        v = scale * (input_tensor / torch.sqrt(squared_norm) + 1e-9)
        return v
