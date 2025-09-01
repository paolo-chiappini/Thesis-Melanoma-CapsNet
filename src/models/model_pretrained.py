import torch
import torch.nn as nn
import torchvision.models as models

from layers import ConvDecoder, MalignancyPredictor, PrimaryCapsules, RoutingCapsules
from utils.layer_output_shape import get_network_output_shape


class ModelPretrainedCapsnet(nn.Module):
    def __init__(
        self,
        img_shape,
        channels,
        num_attributes,
        num_classes,
        primary_dim,
        pose_dim,
        routing_steps,
        pretrained_encoder_path,
        freeze_encoder=True,
        device: torch.device = "cuda",
        routing_algorithm="sigmoid",
    ):
        super(ModelPretrainedCapsnet, self).__init__()
        self.img_shape = img_shape
        self.num_attributes = num_attributes
        self.num_classes = num_classes
        self.device = device

        self.feature_extractor = self._load_pretrained_encoder(
            pretrained_encoder_path, freeze_encoder
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *img_shape)
            feature_map = self.feature_extractor(dummy_input)
            encoder_output_channels = feature_map.shape[1]

        self.primary_capsules = PrimaryCapsules(
            input_channels=encoder_output_channels,
            output_capsules=32,
            output_dim=primary_dim,
            kernel_size=1,  # small kernel size due to Resnet small feature maps
            stride=1,
        )

        with torch.no_grad():
            primary_caps_map = self.primary_capsules(feature_map)
            # Shape will be (batch, num_primary_caps, primary_dim)
            # e.g. (1, 32 * 7 * 7, 8) if the feature map was (1, 512, 7, 7)
            primary_caps_count = primary_caps_map.shape[1]

        # The dimension for the routing output (pose + logit)
        output_dim = pose_dim + 1

        self.attribute_capsules = RoutingCapsules(
            input_dim=primary_dim,
            input_capsules=primary_caps_count,
            num_capsules=num_attributes,
            output_dim=output_dim,
            routing_steps=routing_steps,
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

    def _load_pretrained_encoder(self, checkpoint_path, freeze):
        """
        Loads the pre-trained ResNet18, loads the saved weights,
        decapitates it, and freezes the weights.
        """
        print(f"Loading pre-trained encoder from: {checkpoint_path}")

        full_model = models.resnet18(weights=None)
        num_ftrs = full_model.fc.in_features
        full_model.fc = nn.Linear(num_ftrs, self.num_classes)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        full_model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Successfully loaded weights from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}"
        )

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
        feature_map = self.feature_extractor(x)
        primary_caps_out = self.primary_capsules(feature_map)
        attr_caps_out = self.attribute_capsules(primary_caps_out)
        return attr_caps_out

    def forward(self, x):
        # 1. encode image
        attr_caps_out = self.encode(x)
        # 2. route to attribute capsules
        attribute_logits = attr_caps_out[:, :, 0]
        attribute_poses = attr_caps_out[:, :, 1:]  # shape (N, num_attributes, pose_dim)
        # 3. predict malignancy from poses
        malignancy_scores = self.malignancy_predictor(attribute_poses)

        # 4. reconstuct images from poses
        reconstructions = self.decode(attribute_poses)

        return attribute_logits, reconstructions, malignancy_scores, attribute_poses

    def decode(self, attribute_poses):
        # Removed y_mask (for now...)
        decoder_input = attribute_poses.reshape(attribute_poses.size(0), -1)
        reconstructions = self.decoder(decoder_input)
        reconstructions = reconstructions.reshape(-1, *self.img_shape)
        return reconstructions
