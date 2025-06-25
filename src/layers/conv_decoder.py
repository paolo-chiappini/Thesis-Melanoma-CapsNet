import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(
        self, latent_dim, img_shape, fmap_channels=128, fmap_height=9, fmap_width=9
    ):
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1
            ),  # 9x9 => 18x18
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 18x18 => 36x36
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 36x36 => 72x72
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # 72x72 => 144x144
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                16, 16, kernel_size=4, stride=2, padding=1
            ),  # 144x144 => 288x288
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=0),  # 288x288 => 284x284
            nn.ReLU(inplace=True),
            nn.Conv2d(
                8, self.img_channels, kernel_size=3, stride=1, padding=0
            ),  # 284x284 => 282x282
            nn.Sigmoid(),  # [0, 1] range
        )

    def forward(self, x):
        x = self.fc_layer(x)
        # reshape for convolution
        x = x.view(-1, self.fmap_channels, self.fmap_height, self.fmap_width)
        x = self.decoder(x)
        return x
