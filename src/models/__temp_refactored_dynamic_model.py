import torch
import torch.nn as nn
from layers import (
    Conv2d_BN,
    PrimaryCapsules,
    RoutingCapsules,
    MalignancyPredictor,
    ConvDecoder,
)
from numpy import prod
from utils.layer_output_shape import get_network_output_shape

_PRIMARY_CAPS_IDX = 1


class CapsuleNetworkWithAttributes32(nn.Module):
    """
    Specific modification that aims to work with 32x32 images at primary-caps level
    """

    def __init__(
        self,
        img_shape,  # (H, W, C)
        channels,
        primary_dim,
        num_classes,
        num_attributes,
        output_dim,
        routing_steps,
        device: torch.device,
        kernel_size=3,
        routing_algorithm="softmax",
        min_feature_map=32,  # minimum height/width allowed before stopping downsampling
    ):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        caps_channels = 256

        # -------- Dynamically build encoder -------- #
        self.encoder = self._build_dynamic_encoder(
            channels,
            caps_channels,
            img_shape[0],
            img_shape[1],
            kernel_size,
            min_feature_map,
        )

        # Determine feature map size after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, channels, *img_shape[:2])
            fmap = self.encoder(dummy)
            fmap_channels, fmap_height, fmap_width = fmap.shape[1:]

        # -------- Primary Capsules -------- #
        caps_kernel_size = 9
        caps_stride = 2
        self.primary = PrimaryCapsules(
            fmap_channels,
            fmap_channels,
            primary_dim,
            kernel_size=caps_kernel_size,
            stride=caps_stride,
            padding="valid",
        )

        capsules_output_shape = get_network_output_shape(
            (1, *img_shape), [self.encoder, self.primary], print_all=True
        )
        primary_caps = capsules_output_shape[_PRIMARY_CAPS_IDX]

        # -------- Routing Capsules -------- #
        self.digits = RoutingCapsules(
            primary_dim,
            primary_caps,
            num_attributes,
            output_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        # -------- Classifier -------- #
        self.malignancy_fc = MalignancyPredictor(
            num_attributes=num_attributes,
            capsule_dim=output_dim,
            output_dim=num_classes,
        )

        # -------- Fully dynamic decoder -------- #
        self.decoder_layers = self._build_dynamic_decoder(
            fmap_channels, channels, fmap_height, fmap_width, img_shape[0], img_shape[1]
        )

        self.decoder = ConvDecoder(
            output_dim * num_attributes,
            self.img_shape,
            fmap_channels=fmap_channels,
            fmap_height=fmap_height,
            fmap_width=fmap_width,
            layers=self.decoder_layers,
        )

    def _build_dynamic_encoder(
        self, in_channels, final_channels, height, width, kernel_size, min_fmap
    ):
        """Build encoder that stops downsampling before feature maps become too small."""
        layers = []
        ch = in_channels
        out_ch = 32

        while min(height, width) > min_fmap:
            layers.append(Conv2d_BN(ch, out_ch, kernel_size, stride=2, padding=1))
            ch = out_ch
            out_ch = min(out_ch * 2, final_channels)
            height = (height + 1) // 2
            width = (width + 1) // 2

            # Optional intermediate conv (no downsample)
            layers.append(Conv2d_BN(ch, ch, kernel_size, padding=1))

            if min(height, width) <= min_fmap:
                break

        return nn.Sequential(*layers)

    def _build_dynamic_decoder(
        self, start_channels, out_channels, fmap_h, fmap_w, target_h, target_w
    ):
        """Build decoder that upsamples to match target resolution."""
        layers = []
        in_ch = start_channels
        cur_h, cur_w = fmap_h, fmap_w

        while cur_h < target_h or cur_w < target_w:
            out_ch = max(in_ch // 2, out_channels)
            layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            cur_h *= 2
            cur_w *= 2

        # Adjust to exact size if overshoot
        if cur_h != target_h or cur_w != target_w:
            layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Upsample(
                    size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            )

        # Final conv to output channels
        layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def encode(self, x):
        """
        Encode the image into capsule activations.

        Args:
            x (torch.Tensor): Input image batch.
        Returns:
            torch.Tensor: Capsule activations [B, num_capsules, capsule_dim].
        """
        for layer in self.encoder:
            x = layer(x)
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
