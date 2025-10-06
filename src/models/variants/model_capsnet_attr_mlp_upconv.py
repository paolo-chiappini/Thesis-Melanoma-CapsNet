import torch
import torch.nn as nn

from layers import (
    AttributesPredictor,
    ConvDecoder,
    MalignancyPredictor,
    PrimaryCapsules,
    RoutingCapsules,
)
from layers.conv_batch_norm import Conv2d_BN
from models.model_resnet_classifier import ResnetClassifier


class CapsNetWithAttributesMLPUpconv(nn.Module):
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
        pretrained_encoder_path,
        load_pretrained=False,
        freeze_encoder=True,
        kernel_size=3,
        routing_algorithm="sigmoid",
    ):
        super(CapsNetWithAttributesMLPUpconv, self).__init__()
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

        if load_pretrained:
            self.encoder = self._load_truncated_pretrained_encoder(
                pretrained_encoder_path, freeze_encoder
            )
            encoder_output_channels = 128
        else:
            self.encoder = nn.Sequential(*encoder_layers)
            encoder_output_channels = 256

        # The truncated encoder will output a feature map with 128 channels
        # and a spatial size of 32x32 for a 256x256 input

        self.primary_capsules = PrimaryCapsules(
            input_channels=encoder_output_channels,
            output_channels=caps_channels,
            capsule_dimension=primary_dim,
            kernel_size=9,
            stride=2,
            padding="valid",
        )

        # For a 32x32 input, this layer will produce an output of shape:
        # H_out = floor((32 - 9)/2 + 1) = 12.
        # Total capsules = 32 * 12 * 12 = 4608
        primary_caps_count = 32 * 12 * 12

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

        self.attributes_classifier = AttributesPredictor(capsule_pose_dim=pose_dim)

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

        # squash the output
        # attr_caps_output = squash(attr_caps_output)

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
        attr_caps_outputs = self.encode(x)
        # 2. route to attribute capsules
        attribute_poses = attr_caps_outputs

        # 3. predict malignancy from poses
        malignancy_scores = self.malignancy_predictor(attribute_poses)
        # 4. reconstuct images from poses
        reconstructions = self.decode(attribute_poses, y_mask)

        # 5. reconstruct single-attribute images (with vectorization)
        N, K, pose_dim = attribute_poses.shape

        eye = torch.eye(K, device=self.device).view(1, K, K, 1)
        masked_poses = attribute_poses.unsqueeze(1) * eye
        masked_poses_flat = masked_poses.view(N * K, K, pose_dim)

        attribute_reconstructions_flat = self.decode(masked_poses_flat)
        attribute_reconstructions = attribute_reconstructions_flat.view(
            N, K, *reconstructions.shape[1:]
        )

        # reshape for efficiency
        reshaped_poses = attribute_poses.reshape(N * K, pose_dim)

        logits_flat = self.attributes_classifier(reshaped_poses)
        attribute_logits = logits_flat.view(N, K)

        return {
            "attribute_logits": attribute_logits,
            "reconstructions": reconstructions,
            "attribute_reconstructions": attribute_reconstructions,
            "malignancy_scores": malignancy_scores,
            "attribute_poses": attribute_poses,
        }
