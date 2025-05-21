import torch.nn as nn


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
