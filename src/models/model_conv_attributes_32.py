import torch
import torch.nn as nn
from layers import (
    Conv2d_BN,
    PrimaryCapsules,
    RoutingCapsules,
    MalignancyPredictor,
    ConvDecoder,
)
from numpy import prod
from utils.layer_output_shape import get_network_output_shape


class CapsuleNetworkWithAttributes32(nn.Module):
    """
    Specific modification that aims to work with 32x32 images at primary-caps level
    """

    def __init__(
        self,
        img_shape,
        channels,
        primary_dim,
        num_classes,
        num_attributes,
        output_dim,
        routing_steps,
        device: torch.device,
        kernel_size=3,
        routing_algorithm="softmax",
    ):
        super(CapsuleNetworkWithAttributes32, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        caps_channels = 256

        # Encoder
        encoder_layers = [
            Conv2d_BN(channels, 32, kernel_size, stride=2, padding=1),
            Conv2d_BN(32, 64, kernel_size, padding=1),
            Conv2d_BN(64, 128, kernel_size),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d_BN(128, 192, kernel_size=1, padding=1),
            Conv2d_BN(192, caps_channels, kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Capsules
        caps_kernel_size = 9
        caps_stride = 2

        self.primary = PrimaryCapsules(
            caps_channels,
            caps_channels,
            primary_dim,
            kernel_size=caps_kernel_size,
            stride=caps_stride,
            padding="valid",
        )

        encoder_output_shape = get_network_output_shape(
            (1, *img_shape),  # Add fictitious batch dimension to the input tensor
            encoder_layers,
            print_all=True,
        )
        capsules_output_shape = self.primary.get_output_shape(encoder_output_shape)
        print(f"Capsules output shape: {capsules_output_shape}")
        _, _, h_caps, w_caps = capsules_output_shape
        primary_caps = self.primary._caps_num * h_caps * w_caps

        print(f"Capsules params: {w_caps} X {h_caps} X {primary_caps}")

        self.digits = RoutingCapsules(
            primary_dim,
            primary_caps,
            num_attributes,
            output_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        self.malignancy_fc = MalignancyPredictor(
            num_attributes=num_attributes,
            capsule_dim=output_dim,
            output_dim=num_classes,
        )

        # Decoder
        # decoder_layers = [
        #     nn.Linear(output_dim * num_attributes, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, int(prod(img_shape))),
        #     nn.Sigmoid(),
        # ]
        # self.decoder = nn.Sequential(*decoder_layers)

        self.decoder_layers = [
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = ConvDecoder(
            output_dim * num_attributes,
            self.img_shape,
            fmap_channels=256,
            fmap_height=12,
            fmap_width=12,
            layers=self.decoder_layers,
        )
        encoder_output_shape = get_network_output_shape(
            capsules_output_shape,
            self.decoder_layers,
            print_all=True,
        )

    def encode(self, x):
        """
        Encode the image into capsule activations.

        Args:
            x (torch.Tensor): Input image batch.
        Returns:
            torch.Tensor: Capsule activations [B, num_capsules, capsule_dim].
        """
        for layer in self.encoder:
            x = layer(x)
        out = self.primary(x)
        out = self.digits(out)
        return out

    def classify(self, capsules):
        """
        Return the predictions and malignancy scores

        Args:
            capsules (torch.Tensore): Capsule activations [B, num_capsules, capsule_dim].
        Returns:
            (torch.Tensor, torch.Tensor): (attributes_preds, malignancy_scores).
        """
        malignancy_scores = self.malignancy_fc(capsules)
        preds = torch.norm(capsules, dim=-1)  # Length layer form LaLonde
        return preds, malignancy_scores

    def decode(self, capsules, y=None):
        """
        Returns the reconstruction of input x. If y is None, the max-length capsule is selected.

        Args:
            x (torch.Tensor): Input image batch.
            y (torch.Tensor, optional): One-hot lables for masking capsules. Defaults to None.
        Returns:
            torch.Tensor: Reconstructed images [B, C, H, W].
        """
        preds = torch.norm(capsules, dim=-1)
        if y is None:
            _, max_length_idx = preds.max(dim=1)
            y = torch.eye(self.num_attributes).to(self.device)
            y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        else:
            if y.dim() == 2:
                y = y.unsqueeze(2)
        reconstructions = self.decoder((capsules * y).view(capsules.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)
        return reconstructions

    def forward(self, x, y=None):
        capsules = self.encode(x)
        preds, malignancy_scores = self.classify(capsules)
        reconstructions = self.decode(capsules, y)
        return preds, reconstructions, malignancy_scores, capsules
