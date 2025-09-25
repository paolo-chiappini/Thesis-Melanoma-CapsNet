import torch

from .loss_factory import CombinedLoss


def create_combined_loss(
    config, class_weights=None, attribute_weights=None, device: torch.device = "cuda"
):
    """
    Factory function to create an instance of the CombinedLoss based on configuration.

    Args:
        config (dict): dictionary containing the 'loss' section. Expected key 'components' detailing all loss components.
        class_weights (torch.Tensor, optional): Class weights for malignancy.
        attribute_weights (torch.Tensor, optional): Attribute weights for visual features.

    Returns:
        CombinedLoss: instance of the combined loss components.
    """
    return CombinedLoss(
        config=config,
        class_weights=class_weights,
        attribute_weights=attribute_weights,
        device=device,
    )
