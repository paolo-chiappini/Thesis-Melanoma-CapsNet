import torch
import torch.nn as nn

from layers import (
    AttributesPredictor,
    Conv2d_BN,
    MalignancyPredictor,
    PrimaryCapsules,
    RoutingCapsules,
    SharedFiLMDecoder,
    SimpleDecoder,
)
from utils.layer_output_shape import get_network_output_shape


class CapsNetMultiDecoder(nn.Module):
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
        super(CapsNetMultiDecoder, self).__init__()
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

        self.lesion_decoder = SharedFiLMDecoder(
            num_capsules=num_attributes, pose_dim=pose_dim
        )

        self.per_capsule_decoders = nn.ModuleList(
            [SimpleDecoder(pose_dim=pose_dim) for _ in range(num_attributes)]
        )

        self.attributes_classifier = AttributesPredictor(capsule_pose_dim=pose_dim)

        _ = get_network_output_shape(
            primary_caps_output_shape,
            [self.attribute_capsules],
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

    def decode_all_capsules(self, pose: torch.Tensor) -> torch.Tensor:
        batch_size, K, pose_dim = pose.shape
        reconstructions = []
        for i in range(K):
            reconstructions.append(
                self.per_capsule_decoders[i](pose[:, i, :], out_hw=self.img_shape[1:])
            )
        reconstructions = torch.stack(reconstructions, dim=1)
        return reconstructions

    def decode_lesion(self, attribute_poses: torch.Tensor) -> torch.Tensor:
        decoder_input = attribute_poses.reshape(attribute_poses.size(0), -1)
        reconstructions = self.lesion_decoder(decoder_input, out_hw=self.img_shape[1:])
        reconstructions = reconstructions.reshape(-1, *self.img_shape)
        return reconstructions

    def decode(self, attribute_poses: torch.Tensor) -> torch.Tensor:
        return self.decode_lesion(attribute_poses), self.decode_all_capsules(
            attribute_poses
        )

    def forward(self, x):
        # 1. encode image
        attr_caps_outputs = self.encode(x)
        # 2. route to attribute capsules
        attribute_poses = attr_caps_outputs

        # 3. predict malignancy from poses
        malignancy_scores = self.malignancy_predictor(attribute_poses)
        # 4. reconstuct images from poses
        reconstructions, attribute_reconstructions = self.decode(attribute_poses)
        N, K, pose_dim = attribute_poses.shape

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
