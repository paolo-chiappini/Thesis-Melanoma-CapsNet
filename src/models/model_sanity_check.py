import torch
import torch.nn as nn
from torchvision import models


class SanityCheckModel(nn.Module):
    def __init__(self, img_shape, num_classes=7, device="cuda"):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        # replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.backbone(x)
