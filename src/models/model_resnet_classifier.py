import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResnetClassifier(nn.Module):

    def __init__(self, num_classes: int = 1, pretrained: bool = True, **kwargs):
        super(ResnetClassifier, self).__init__()

        self.weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=self.weights)

        self.preprocess = (
            self.weights.transforms()
            if pretrained
            else models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        )

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.preprocess(x)
        out = self.backbone(x)

        features = torch.flatten(out, 1)
        logits = self.fc(features)

        return {"malignancy_scores": logits, "image_features": features}
