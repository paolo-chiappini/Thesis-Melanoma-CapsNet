import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResnetClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        droupout_rate: float = 0.5,
        pretrained: bool = True,
        **kwargs
    ):
        super(ResnetClassifier, self).__init__()

        self.weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnet18(weights=self.weights)

        self.preprocess = (
            self.weights.transforms()
            if pretrained
            else models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        )

        num_filters = self.encoder.fc.in_features

        self.encoder.fc = nn.Sequential(
            nn.Dropout(p=droupout_rate), nn.Linear(num_filters, num_classes)
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.preprocess(x)
        out = self.encoder(x)
        return {"malignancy_scores": out}

    def get_feature_extractor(self) -> nn.Module:
        """
        Returns:
            nn.Module: decapitated version of the model (all layers except avgpool and fc).
        """
        feature_extractor = nn.Sequential(*list(self.encoder.children())[:-2])
        return feature_extractor
