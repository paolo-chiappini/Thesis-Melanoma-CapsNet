import torch
import torch.nn as nn
import torchvision.models as models

from layers import (
    AttributesPredictor,
    ConvDecoder,
    MalignancyPredictor,
    PrimaryCapsules,
    RoutingCapsules,
)
from utils.commons import strip_module_prefix
from utils.layer_output_shape import get_network_output_shape


class CapsNetWithAttributesMLP_Pre(nn.Module):
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
        freeze_encoder=True,
        kernel_size=3,
        routing_algorithm="sigmoid",
    ):
        super(CapsNetWithAttributesMLP_Pre, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.device = device
        caps_channels = 256

        self.encoder = self._load_pretrained_encoder(
            pretrained_encoder_path, freeze_encoder
        )

        # Capsules
        with torch.no_grad():
            dummy_input = torch.zeros(1, *img_shape)
            feature_map = self.encoder(dummy_input)
            encoder_output_channels = feature_map.shape[1]

        self.primary_capsules = PrimaryCapsules(
            encoder_output_channels,
            caps_channels,
            primary_dim,
            kernel_size=1,
            stride=2,
            padding="valid",
        )

        primary_caps_output_shape = get_network_output_shape(
            feature_map.shape, [self.primary_capsules]
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

    def _load_pretrained_encoder(self, checkpoint_path, freeze):
        """
        Loads the pre-trained ResNet18, loads the saved weights,
        decapitates it, and freezes the weights.
        """
        print(f"Loading pre-trained encoder from: {checkpoint_path}")

        full_model = models.resnet18(weights=None)
        num_ftrs = full_model.fc.in_features
        full_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        full_model.load_state_dict(
            strip_module_prefix(checkpoint, prefix="module.encoder."),
        )
        print("Successfully loaded weights")

        # We take all layers except for the final adaptive average pooling and the fully connected layer.
        feature_extractor = nn.Sequential(*list(full_model.children())[:-2])

        if freeze:
            print("Freezing encoder weights.")
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            print("Encoder weights are NOT frozen. Fine-tuning enabled.")

        return feature_extractor

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
        attr_caps_outputs = self.encode(x)
        # 2. route to attribute capsules
        attribute_poses = attr_caps_outputs

        # 3. predict malignancy from poses
        malignancy_scores = self.malignancy_predictor(attribute_poses)
        # 4. reconstuct images from poses
        reconstructions = self.decode(attribute_poses, y_mask)

        N, K, pose_dim = attribute_poses.shape
        # reshape for efficiency
        reshaped_poses = attribute_poses.reshape(N * K, pose_dim)

        logits_flat = self.attributes_classifier(reshaped_poses)
        attribute_logits = logits_flat.view(N, K)

        # return attribute_logits, reconstructions, malignancy_scores, attribute_poses
        return {
            "attribute_logits": attribute_logits,
            "reconstructions": reconstructions,
            "malignancy_scores": malignancy_scores,
            "attribute_poses": attribute_poses,
        }
