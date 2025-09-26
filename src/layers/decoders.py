from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_batch_norm import Conv2d_BN


def UpconvBlock(
    in_channels: int,
    out_channels: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        Conv2d_BN(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        ),
    )


class SimpleDecoder(nn.Module):
    """
    Simple small decoder that upsamples from a latent vector to an image used for single-pose reconstructions.
    """

    def __init__(
        self, pose_dim: int = 32, base_channels: int = 128, out_size: int = 256
    ):
        super(SimpleDecoder, self).__init__()
        self.pose_dim = pose_dim
        self.out_size = out_size
        self.init_spatial_size = 8
        self.init_channels = base_channels

        self.fc = nn.Linear(
            pose_dim,
            self.init_channels * self.init_spatial_size * self.init_spatial_size,
        )
        self.decode = nn.Sequential(
            UpconvBlock(self.init_channels, base_channels // 2),
            UpconvBlock(base_channels // 2, base_channels // 4),
            UpconvBlock(base_channels // 4, base_channels // 8),
            UpconvBlock(base_channels // 8, base_channels // 16),
            UpconvBlock(base_channels // 16, base_channels // 32),
            nn.Conv2d(base_channels // 32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, out_hw: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): capsule pose.
            out_hw (Tuple[int, int], optional): output size. Defaults to None.
        """
        x = self.fc(z)
        batch_size = x.shape[0]
        x = x.view(
            batch_size,
            self.init_channels,
            self.init_spatial_size,
            self.init_spatial_size,
        )
        x = self.decode(x)
        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class FiLMBlock(nn.Module):
    """
    Shared FiLM decoder block for global reconstructions.
    """

    def __init__(self, channels: int, film_dim: int):
        super(FiLMBlock, self).__init__()
        self.channels = channels

        self.scale = nn.Linear(film_dim, channels)
        self.shift = nn.Linear(film_dim, channels)

    def forward(self, x: torch.Tensor, film_vec: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(film_vec).view(-1, self.channels, 1, 1)
        beta = self.shift(film_vec).view(-1, self.channels, 1, 1)
        return x * (1 + gamma) + beta


class SharedFiLMDecoder(nn.Module):
    """
    Shared decoder for global reconstructions.
    """

    def __init__(
        self,
        num_capsules: int = 8,
        pose_dim: int = 32,
        base_channels: int = 128,
        out_size: int = 256,
        film_dim: int = 32,
    ):
        super(SharedFiLMDecoder, self).__init__()
        self.out_size = out_size

        self.film_mlp = nn.Sequential(
            nn.Linear(pose_dim * num_capsules, film_dim),
            nn.ReLU(),
            nn.Linear(film_dim, film_dim),
        )

        self.fc = nn.Linear(pose_dim * num_capsules, base_channels * 4 * 4)

        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.film1 = FiLMBlock(base_channels, film_dim)
        self.up1 = UpconvBlock(base_channels, base_channels // 2)
        self.film2 = FiLMBlock(base_channels // 2, film_dim)
        self.up2 = UpconvBlock(base_channels // 2, base_channels // 4)
        self.film3 = FiLMBlock(base_channels // 4, film_dim)
        self.up3 = UpconvBlock(base_channels // 4, base_channels // 8)
        self.film4 = FiLMBlock(base_channels // 8, film_dim)
        self.up4 = UpconvBlock(base_channels // 8, base_channels // 16)
        self.film5 = FiLMBlock(base_channels // 16, film_dim)
        self.up5 = UpconvBlock(base_channels // 16, base_channels // 32)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_channels // 32, 3, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(
        self, pose: torch.Tensor, out_hw: Tuple[int, int] = None
    ) -> torch.Tensor:
        film_vec = self.film_mlp(pose)

        x = self.fc(pose).view(pose.size(0), -1, 4, 4)
        x = F.relu(self.conv1(x))
        x = self.film1(x, film_vec)
        x = self.up1(x)
        x = self.film2(x, film_vec)
        x = self.up2(x)
        x = self.film3(x, film_vec)
        x = self.up3(x)
        x = self.film4(x, film_vec)
        x = self.up4(x)
        x = self.film5(x, film_vec)
        x = self.up5(x)

        x = self.to_rgb(x)

        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x
