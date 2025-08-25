import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class TransferModel(nn.Module):

    def __init__(self, num_classes=2, **kwargs):
        super(TransferModel, self).__init__()

        self.weights = ResNet34_Weights.DEFAULT
        self.encoder = models.resnet34(weights=self.weights)

        self.preprocess = self.weights.transforms()

        num_filters = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.preprocess(x)
        out = self.encoder(x)
        return out
