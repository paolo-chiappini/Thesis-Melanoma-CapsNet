import torch
import torch.nn as nn
from layers import ConvDecoder


class ConvAutoencoder(nn.Module):
    def __init__(self, image_shape=(3, 282, 282), latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.image_channels = self.image_shape[0]

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.image_channels, 8, kernel_size=4, stride=2, padding=1
            ),  # 282 -> 141
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # 141 -> 70
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 70 -> 35
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 35 -> 17
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 17 -> 8
            nn.ReLU(inplace=True),
        )

        self.fc_enc = nn.Linear(128 * 8 * 8, latent_dim)

        # --- DECODER ---
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            img_shape=self.image_shape,
            fmap_height=9,
            fmap_width=9,
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
