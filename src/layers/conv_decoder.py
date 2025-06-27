import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        img_shape,
        fmap_channels=128,
        fmap_height=9,
        fmap_width=9,
        layers=None,
    ):
        self.layers = (
            [
                nn.ConvTranspose2d(
                    512, 256, kernel_size=4, stride=2, padding=1
                ),  # 8 -> 16
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    256, 128, kernel_size=4, stride=2, padding=1
                ),  # 16 -> 32
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128, 64, kernel_size=4, stride=2, padding=1
                ),  # 32 -> 64
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    64, 32, kernel_size=4, stride=2, padding=1
                ),  # 64 -> 128
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    32, 16, kernel_size=4, stride=2, padding=1
                ),  # 64 -> 128
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    16, self.img_channels, kernel_size=3, stride=1, padding=1
                ),  # 128 -> 128
                nn.Sigmoid(),  # [0, 1] range
            ]
            if layers is None
            else layers
        )

        super().__init__()
        self.img_channels = img_shape[0]
        self.img_height = img_shape[1]
        self.img_width = img_shape[1]

        self.fmap_channels = fmap_channels
        self.fmap_height = fmap_height
        self.fmap_width = fmap_width

        self.fc_layer = nn.Linear(
            latent_dim, self.fmap_channels * self.fmap_height * self.fmap_width
        )

        self.decoder = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.fc_layer(x)
        # reshape for convolution
        x = x.view(-1, self.fmap_channels, self.fmap_height, self.fmap_width)
        x = self.decoder(x)
        return x
