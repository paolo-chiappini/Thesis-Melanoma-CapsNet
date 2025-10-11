import torch
import torch.nn as nn

from layers import (
    AttributesPredictor,
    ConstrainedRoutingCapsules,
    ConvDecoder,
    MalignancyPredictor,
    MaskDecoderHead,
    PrimaryCapsules,
)
from layers.conv_batch_norm import Conv2d_BN
from layers.routing_capsules import RoutingCapsules
from models.model_resnet_classifier import ResnetClassifier


class CapsNetWithAttributesMLPUpconv(nn.Module):
    def __init__(
        self,
        img_shape: tuple,
        channels: int,
        num_attributes: int,
        num_classes: int,
        primary_dim: int,
        pose_dim: int,
        routing_steps: int,
        device: torch.device,
        pretrained_encoder_path: str,
        load_pretrained: bool = False,
        freeze_encoder: bool = True,
        kernel_size: int = 3,
        routing_algorithm: str = "softmax",
    ):
        super(CapsNetWithAttributesMLPUpconv, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        self.channels = channels
        caps_channels = 256

        # determine encoder output structure
        if load_pretrained:
            self.encoder = self._load_truncated_pretrained_encoder(
                pretrained_encoder_path, freeze_encoder
            )
            encoder_output_channels = 128
        else:
            self.encoder = nn.Sequential(
                [
                    Conv2d_BN(channels, 32, kernel_size, stride=2, padding=1),
                    Conv2d_BN(32, 64, kernel_size, padding=1),
                    Conv2d_BN(64, 128, kernel_size),
                    nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                    Conv2d_BN(128, 192, kernel_size=1, padding=1),
                    Conv2d_BN(192, caps_channels, kernel_size, padding=1),
                    nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                ]
            )
            encoder_output_channels = 256

        self.primary_capsules = PrimaryCapsules(
            input_channels=encoder_output_channels,
            output_channels=caps_channels,
            capsule_dimension=primary_dim,
            kernel_size=9,
            stride=2,
            padding="valid",
        )

        # total_caps = (caps_channels / primary_dim) * 12 * 12
        # assuming primary_dim = 8 => 32 * 12 * 12 = 4608
        primary_caps_count = (caps_channels // primary_dim) * 12 * 12

        self.attribute_capsules = RoutingCapsules(
            primary_dim,
            primary_caps_count,
            num_attributes,
            pose_dim,
            routing_steps,
            device=self.device,
            routing_algorithm=routing_algorithm,
        )

        self.attribute_capsules = ConstrainedRoutingCapsules(
            num_primary_caps=primary_caps_count,
            num_attribute_caps=num_attributes,
            primary_dim=primary_dim,
            pose_dim=pose_dim,
            num_iterations=routing_steps,
        )

        self.malignancy_predictor = MalignancyPredictor(
            num_attributes=num_attributes,
            capsule_dim=pose_dim,
            output_dim=num_classes,
        )

        self.attributes_classifier = AttributesPredictor(capsule_pose_dim=pose_dim)

        self.decoder_layers = [
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # 192x192
            nn.Upsample(scale_factor=1.3334, mode="bilinear"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
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

        self.mask_decoder = MaskDecoderHead(
            num_attributes=num_attributes, pose_dim=pose_dim, output_size=self.img_shape
        )

    def _load_truncated_pretrained_encoder(
        self, checkpoint_path: str, freeze: bool
    ) -> nn.Module:
        print(f"Loading pre-trained encoder from: {checkpoint_path}")

        full_trained_model = ResnetClassifier()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        full_trained_model.load_state_dict(checkpoint)
        print("Successfully loaded weights")

        children = list(full_trained_model.backbone.children())
        truncated_backbone = nn.Sequential(*children[:6])

        if freeze:
            print("Freezing encoder weights.")
            for param in truncated_backbone.parameters():
                param.requires_grad = False
        else:
            print("Encoder weights are NOT frozen. Fine-tuning enabled.")

        return truncated_backbone

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the image into capsule activations.

        Args:
            x (torch.Tensor): Input image batch.
        Returns:
            torch.Tensor: Capsule activations [B, num_capsules, capsule_dim].
        """
        encoded_features = self.encoder(x)
        primary_caps_output = self.primary_capsules(encoded_features)
        attr_caps_poses, coupling_coefficients = self.attribute_capsules(
            primary_caps_output
        )
        return attr_caps_poses, coupling_coefficients

    def decode(
        self, attribute_poses: torch.Tensor, y_mask: torch.Tensor = None
    ) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. encode image into poses
        attribute_poses, coupling_coefficients = self.encode(x)
        N, K, pose_dim = attribute_poses.shape

        # 2. predict malignancy
        malignancy_scores = self.malignancy_predictor(attribute_poses)

        # 3. reconstruct full image
        reconstructions = self.decode(attribute_poses, y_mask)

        # 4. predict attribute classifications
        reshaped_poses = attribute_poses.reshape(N * K, pose_dim)
        logits_flat = self.attributes_classifier(reshaped_poses)
        attribute_logits = logits_flat.view(N, K)

        # 5. reconstruct single-attribute images
        eye = torch.eye(K, device=self.device).view(1, K, K, 1)
        masked_poses = attribute_poses.unsqueeze(1) * eye
        masked_poses_flat = masked_poses.view(N * K, K, pose_dim)

        attribute_reconstructions_flat = self.decode(masked_poses_flat)
        attribute_reconstructions = attribute_reconstructions_flat.view(
            N, K, *reconstructions.shape[1:]
        )

        # 6. predict attribute masks
        predicted_masks = self.mask_decoder(attribute_poses)

        return {
            "attribute_logits": attribute_logits,
            "reconstructions": reconstructions,
            "attribute_reconstructions": attribute_reconstructions,
            "malignancy_scores": malignancy_scores,
            "attribute_poses": attribute_poses,
            "predicted_masks": predicted_masks,
            "coupling_coefficients": coupling_coefficients,
        }
