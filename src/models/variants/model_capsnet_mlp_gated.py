import torch
import torch.nn as nn

from layers import (
    AttributesPredictor,
    Conv2d_BN,
    ConvDecoder,
    MalignancyPredictor,
    PrimaryCapsules,
    RoutingCapsules,
)
from utils.layer_output_shape import get_network_output_shape


class CapsNetWithAttributesMLPGated(nn.Module):
    def __init__(
        self,
        img_shape,
        channels,
        primary_dim,
        num_classes,
        num_attributes,
        pose_dim,
        routing_steps,
        device: torch.device,
        kernel_size=3,
        routing_algorithm="sigmoid",
    ):
        super(CapsNetWithAttributesMLPGated, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        caps_channels = 256

        # Encoder
        encoder_layers = [
            Conv2d_BN(channels, 32, kernel_size, stride=2, padding=1),
            Conv2d_BN(32, 64, kernel_size, padding=1),
            Conv2d_BN(64, 128, kernel_size),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d_BN(128, 192, kernel_size=1, padding=1),
            Conv2d_BN(192, caps_channels, kernel_size, padding=1),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Capsules
        primary_caps_input_shape = get_network_output_shape(
            (1, *img_shape), encoder_layers
        )
        self.primary_capsules = PrimaryCapsules(
            primary_caps_input_shape[1],
            caps_channels,
            primary_dim,
            kernel_size=9,
            stride=2,
            padding="valid",
        )

        primary_caps_output_shape = get_network_output_shape(
            (1, *img_shape), [*encoder_layers, self.primary_capsules], print_all=True
        )
        primary_caps_count = primary_caps_output_shape[1]

        output_dim = pose_dim

        self.attribute_capsules = RoutingCapsules(
            primary_dim,
            primary_caps_count,
            num_attributes,
            output_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        self.malignancy_predictor = MalignancyPredictor(
            num_attributes=num_attributes,
            capsule_dim=pose_dim,
            output_dim=num_classes,
        )

        self.decoder_layers = [
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = ConvDecoder(
            pose_dim * num_attributes,
            self.img_shape,
            fmap_channels=256,
            fmap_height=12,
            fmap_width=12,
            layers=self.decoder_layers,
        )

        self.attributes_classifier = AttributesPredictor(capsule_pose_dim=pose_dim)

        attr_caps_output_dim = get_network_output_shape(
            primary_caps_output_shape,
            [self.attribute_capsules],
            print_all=True,
        )

        attr_caps_output_dim = list(attr_caps_output_dim)
        _ = get_network_output_shape(
            attr_caps_output_dim,
            [self.decoder],
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
        encoded_features = self.encoder(x)
        primary_caps_output = self.primary_capsules(encoded_features)
        attr_caps_output = self.attribute_capsules(primary_caps_output)
        return attr_caps_output

    def decode(self, attribute_poses, y_mask=None):
        """
        Returns the reconstruction of input x. If y is None, the max-length capsule is selected.

        Args:
            x (torch.Tensor): Input image batch.
            y (torch.Tensor, optional): One-hot lables for masking capsules. Defaults to None.
        Returns:
            torch.Tensor: Reconstructed images [B, C, H, W].
        """
        if y_mask is not None:
            if y_mask.dim() == 2:
                y_mask = y_mask.unsqueeze(2)  # reshape for broadcasting
            masked_poses = attribute_poses * y_mask
        else:
            masked_poses = attribute_poses

        decoder_input = masked_poses.reshape(masked_poses.size(0), -1)
        reconstructions = self.decoder(decoder_input)
        reconstructions = reconstructions.reshape(-1, *self.img_shape)
        return reconstructions

    def forward(self, x, y_mask=None):
        # 1. encode image
        attr_caps_outputs = self.encode(x)  # (N, K, d)

        N, K, pose_dim = attr_caps_outputs.shape  # unpack here

        attribute_poses = attr_caps_outputs

        # 2. predict malignancy scores
        malignancy_scores = self.malignancy_predictor(attribute_poses)

        # 3. predict attribute logits from poses
        reshaped_poses = attribute_poses.reshape(N * K, pose_dim)  # flatten capsules

        logits_flat = self.attributes_classifier(reshaped_poses)  # (N * K, 1)
        attribute_logits = logits_flat.view(N, K)  # (N, K)

        # 4. soft gating with detach
        p_gate = torch.sigmoid(attribute_logits)  # soft gate
        p_gate_detached = p_gate.detach()
        attribute_poses_gated = attribute_poses * p_gate_detached.unsqueeze(
            -1
        )  # (N, K, d)

        # 5. decode gated poses to reconstruction
        reconstructions = self.decode(attribute_poses_gated, y_mask)

        # 6. reconstruct single-attribute images vectorized
        eye = torch.eye(K, device=attribute_poses.device).view(
            1, K, K, 1
        )  # (1, K, K, 1)
        masked_poses = attribute_poses_gated.unsqueeze(1) * eye  # (N, K, K, d)
        masked_poses_flat = masked_poses.view(
            N * K, K, pose_dim
        )  # flatten for batch decode

        attribute_reconstructions_flat = self.decode(masked_poses_flat)
        attribute_reconstructions = attribute_reconstructions_flat.view(
            N, K, *reconstructions.shape[1:]
        )

        return {
            "attribute_logits": attribute_logits,
            "reconstructions": reconstructions,
            "attribute_reconstructions": attribute_reconstructions,
            "malignancy_scores": malignancy_scores,
            "attribute_poses": attribute_poses,
        }
