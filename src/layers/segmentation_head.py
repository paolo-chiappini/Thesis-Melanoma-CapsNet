import torch
import torch.nn as nn
import torch.functional as F


class SegmentationHead(nn.Module):
    def __init__(self, capsule_dim, num_capsules, output_size):
        super(SegmentationHead, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                num_capsules, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )
        self.output_size = output_size

    def forward(self, capsule_activations):
        x = capsule_activations
        x = self.deconv(x)
        return F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )
