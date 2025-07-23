import torch
import torch.nn as nn
from layers import ConvDecoder


class ConvAutoencoder(nn.Module):
    def __init__(self, img_shape=(3, 282, 282), latent_dim=1024, device="cuda"):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = img_shape
        self.image_channels = self.image_shape[0]
        self.device = device

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.image_channels, 16, kernel_size=3, stride=2, padding=1
            ),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
        )

        self.fc_enc = nn.Linear(512 * 8 * 8, latent_dim)

        # --- DECODER ---
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            img_shape=self.image_shape,
            fmap_channels=512,
            fmap_height=8,
            fmap_width=8,
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
