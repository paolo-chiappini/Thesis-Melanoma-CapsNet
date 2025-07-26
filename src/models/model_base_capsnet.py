import torch
import torch.nn as nn
from layers import PrimaryCapsules, RoutingCapsules, MalignancyPredictor, ConvDecoder
from utils.layer_output_shape import get_network_output_shape

_PRIMARY_CAPS_IDX = 1


class CapsuleNetworkBase(nn.Module):
    def __init__(
        self,
        img_shape,
        channels,
        primary_dim,
        num_classes,
        num_attributes,
        output_dim,
        routing_steps,
        device: torch.device,
        kernel_size=9,
        routing_algorithm="softmax",
    ):
        super(CapsuleNetworkBase, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        caps_channels = 256

        self.conv1 = nn.Conv2d(
            img_shape[0], caps_channels, kernel_size=kernel_size, stride=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        # Capsules
        caps_kernel_size = 9
        caps_stride = 3

        self.primary = PrimaryCapsules(
            caps_channels,
            caps_channels,
            primary_dim,
            kernel_size=caps_kernel_size,
            stride=caps_stride,
            padding="valid",
        )

        capsules_output_shape = get_network_output_shape(
            (1, *img_shape), [self.conv1, self.relu, self.primary], print_all=True
        )
        primary_caps = capsules_output_shape[_PRIMARY_CAPS_IDX]

        self.digits = RoutingCapsules(
            primary_dim,
            primary_caps,
            num_attributes,
            output_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        self.malignancy_fc = MalignancyPredictor(
            num_attributes=num_attributes,
            capsule_dim=output_dim,
            output_dim=num_classes,
        )

        self.decoder_layers = [
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=img_shape[1:], mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = ConvDecoder(
            output_dim * num_attributes,
            self.img_shape,
            fmap_channels=256,
            fmap_height=12,
            fmap_width=12,
            layers=nn.Sequential(*self.decoder_layers),
        )

        _ = get_network_output_shape(
            capsules_output_shape,
            [self.digits, self.decoder],
            print_all=True,
        )

    def encode(self, x):
        """
        Encode the image into capsule activations.

        Args:
            x (torch.Tensor): Input image batch.
        Returns:
            torch.Tensor: Capsule activations [B, num_capsules, capsule_dim].
        """
        x = self.conv1(x)
        x = self.relu(x)
        out = self.primary(x)
        out = self.digits(out)
        return out

    def classify(self, capsules):
        """
        Return the predictions and malignancy scores

        Args:
            capsules (torch.Tensore): Capsule activations [B, num_capsules, capsule_dim].
        Returns:
            (torch.Tensor, torch.Tensor): (attributes_preds, malignancy_scores).
        """
        malignancy_scores = self.malignancy_fc(capsules)
        preds = torch.norm(capsules, dim=-1)  # Length layer form LaLonde
        return preds, malignancy_scores

    def decode(self, capsules, y=None):
        """
        Returns the reconstruction of input x. If y is None, the max-length capsule is selected.

        Args:
            x (torch.Tensor): Input image batch.
            y (torch.Tensor, optional): One-hot lables for masking capsules. Defaults to None.
        Returns:
            torch.Tensor: Reconstructed images [B, C, H, W].
        """
        preds = torch.norm(capsules, dim=-1)
        if y is None:
            _, max_length_idx = preds.max(dim=1)
            y = torch.eye(self.num_attributes).to(self.device)
            y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        else:
            if y.dim() == 2:
                y = y.unsqueeze(2)
        reconstructions = self.decoder((capsules * y).view(capsules.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)
        return reconstructions

    def forward(self, x, y=None):
        capsules = self.encode(x)
        preds, malignancy_scores = self.classify(capsules)
        reconstructions = self.decode(capsules, y)
        return preds, reconstructions, malignancy_scores, capsules
