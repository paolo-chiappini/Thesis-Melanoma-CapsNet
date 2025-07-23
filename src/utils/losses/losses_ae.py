import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        resize=True,
        resize_size=(224, 244),
        model=vgg16(pretrained=True).features[:16],
    ):
        super(PerceptualLoss, self).__init__()
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.resize = resize
        self.resize_size = resize_size

    def forward(self, input, target):
        if self.resize:
            input = F.interpolate(
                input, size=self.resize_size, mode="bilinear", align_corners=False
            )
            target = F.interpolate(
                input, size=self.resize_size, mode="bilinear", align_corners=False
            )
        input_model = self.model(input)
        target_model = self.model(target)
        return F.l1_loss(input_model, target_model)


class AECompositeLoss(nn.Module):
    def __init__(
        self, alpha=1.0, beta=1.0, gamma=0.1, perceptual=True, class_weights=None
    ):
        super(AECompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.perceptual = PerceptualLoss() if perceptual else None
        self.l1 = nn.L1Loss()
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def forward(self, reconstruction, target):
        l1_loss = self.l1(reconstruction, target)

        if self.class_weights is not None:
            # ensure class_weights matches the number of channels
            cw = self.class_weights.to(reconstruction.device)
            cw = cw.view(1, -1, 1, 1)  # shape for broadcasting
            l1_loss = l1_loss * cw

        ssim_loss = 1 - ssim(reconstruction, target, data_range=1.0, size_average=True)
        perceptual_loss = (
            self.perceptual(reconstruction, target) if self.perceptual else 0.0
        )

        total = (
            self.alpha * l1_loss + self.beta * ssim_loss + self.gamma * perceptual_loss
        )
        return total
