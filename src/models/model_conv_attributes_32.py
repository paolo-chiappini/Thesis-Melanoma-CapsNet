import torch
import torch.nn as nn
from layers import Conv2d_BN, PrimaryCapsules, RoutingCapsules, MalignancyPredictor
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
            Conv2d_BN(channels, 32, kernel_size, stride=2, padding="valid"),
            Conv2d_BN(32, 64, kernel_size, padding="valid"),
            Conv2d_BN(64, 128, kernel_size),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d_BN(128, 192, kernel_size=1, padding="valid"),
            Conv2d_BN(192, caps_channels, kernel_size, padding="valid"),
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
        decoder_layers = [
            nn.Linear(output_dim * num_attributes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(prod(img_shape))),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        out = self.primary(x)
        out = self.digits(out)

        malignancy_scores = self.malignancy_fc(out)
        preds = torch.norm(out, dim=-1)  # Length layer form LaLonde

        # Reconstruct image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_attributes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)

        return preds, reconstructions, malignancy_scores
