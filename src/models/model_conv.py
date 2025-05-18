import torch
import torch.nn as nn
from . import capsules as caps
from numpy import prod


class Conv2d_BN(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size=3, stride=1, padding="same"
    ):
        super(Conv2d_BN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


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

        # (192, 32, 32) -> (192, 16, 16)
        self.conv2d_6 = nn.Conv2d(
            192, caps_channels, kernel_size=kernel_size, stride=2, padding="valid"
        )

        # Capsules
        self.primary = caps.PrimaryCapsules(caps_channels, caps_channels, primary_dim)

        caps_kernel_size = 9
        # primary_caps = int(
        #     caps_channels
        #     / primary_dim
        #     * (img_shape[1] - 2 * (caps_kernel_size - 1))
        #     * (img_shape[2] - 2 * (caps_kernel_size - 1))
        #     / 4
        # )
        # primary_caps = int(caps_channels / primary_dim * (16**2))
        primary_caps = 512  # 4 x 4 feature maps for 32 capsules
        self.digits = caps.RoutingCapsules(
            primary_dim,
            primary_caps,
            num_classes,
            output_dim,
            routing_steps,
            device=self.device,
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
        preds = torch.norm(out, dim=-1)

        # Reconstruct image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)

        return preds, reconstructions
