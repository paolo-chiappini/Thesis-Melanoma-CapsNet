import torch
import torch.nn as nn


class MaskDecoderHead(nn.Module):

    def __init__(self, num_attributes: int, pose_dim: int, output_size: tuple):
        super(MaskDecoderHead, self).__init__()
        self.num_attributes = num_attributes
        _, self.H, self.W = output_size

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pose_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, self.H * self.W),
                    nn.Sigmoid(),
                )
                for _ in range(num_attributes)
            ]
        )

    def forward(self, attribute_poses: torch.Tensor) -> torch.Tensor:
        B, K, pose_dim = attribute_poses.shape
        predicted_masks = []

        for k in range(K):
            pose = attribute_poses[:, k, :]
            mask_flat = self.decoders[k](pose)
            mask = mask_flat.view(B, 1, self.H, self.W)
            predicted_masks.append(mask)

        return torch.cat(predicted_masks, dim=1)
