import torch.nn as nn
from utils.functional import squash
from utils.layer_output_shape import get_conv2d_output_shape


class PrimaryCapsules(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        capsule_dimension,
        kernel_size=9,
        stride=2,
        padding="valid",
    ):
        """
        Params:
        - input_channels:   number of channels in input.
        - output_channles:  number of channels in output.
        - capsule_dimension:         dimension (length) of capsule output vector.
        """
        super(PrimaryCapsules, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.capsule_dimension = capsule_dimension
        self._caps_num = int(output_channels / capsule_dimension)  # number of capsules

        assert self._caps_num > 0, "Number of capsules must be greater than 0"

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        out = self.conv(x)
        # Reshape output (batch_size, _caps_num, H_caps, W_caps, C_caps)
        out = out.view(
            out.size(0),
            self._caps_num,
            out.size(2),
            out.size(3),
            self.capsule_dimension,
        )
        # Flatten outpus into (batch_size, _caps_num * H_caps * W_caps, C_caps)
        out = out.view(out.size(0), -1, self.capsule_dimension)
        return squash(out)

    def get_output_shape(self, input_shape):
        return get_conv2d_output_shape(
            input_shape,
            self.output_channels,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.padding,
        )
