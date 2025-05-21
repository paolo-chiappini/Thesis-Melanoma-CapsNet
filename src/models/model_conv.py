import torch
import torch.nn as nn
from numpy import prod
from layers import Conv2d_BN
from layers import PrimaryCapsules, RoutingCapsules


class CapsuleNetwork(nn.Module):
    """
    Network architecture from Pérez and Ventura DOI: 10.3390/cancers13194974
    """

    def __init__(
        self,
        img_shape,
        channels,
        primary_dim,
        num_classes,
        output_dim,
        routing_steps,
        device: torch.device,
        kernel_size=3,
        routing_algorithm="softmax",
    ):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device
        caps_channels = 256

        # Encoder
        # (3, 270, 270) -> (32, 134, 134)
        self.conv1 = Conv2d_BN(channels, 32, kernel_size, stride=2, padding="valid")
        # (32, 134, 134) -> (32, 132, 132)
        self.conv2 = Conv2d_BN(32, 32, kernel_size, padding="valid")
        # (32, 132, 132) -> (64, 132, 132)
        self.conv3 = Conv2d_BN(32, 64, kernel_size)

        # (64, 132, 132) -> (64, 66, 66)
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=2)

        # (64, 66, 66) -> (80, 66, 66)
        self.conv4 = Conv2d_BN(64, 80, kernel_size=1, padding="valid")
        # (80, 66, 66) -> (192, 64, 64)
        self.conv5 = Conv2d_BN(80, 192, kernel_size, padding="valid")

        # (192, 64, 64) -> (192, 32, 32)
        self.max_pooling2d_2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2)

        # (192, 32, 32) -> (256, 16, 16)
        self.conv2d_6 = nn.Conv2d(
            192, caps_channels, kernel_size=kernel_size, stride=2, padding="valid"
        )

        # Capsules
        caps_kernel_size = 9
        caps_stride = 2
        # (256, 16, 16) -> (256, 4, 4)
        self.primary = PrimaryCapsules(
            caps_channels,
            caps_channels,
            primary_dim,
            kernel_size=caps_kernel_size,
            stride=caps_stride,
            padding="valid",
        )

        primary_caps = 512  # 256 * 4 * 4

        self.digits = RoutingCapsules(
            primary_dim,
            primary_caps,
            num_classes,
            output_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(prod(img_shape))),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.max_pooling2d_1(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.max_pooling2d_2(out)
        out = self.conv2d_6(out)
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)  # Length layer form LaLonde

        # Reconstruct image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)

        return preds, reconstructions
