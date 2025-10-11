import torch


def get_primary_caps_coords(
    img_size: int,
    encoder_stride: int = 8,
    primary_kernel: int = 9,
    primary_stride: int = 2,
) -> torch.Tensor:
    """
    Computes the center coordinates of all primary capsules in the original image space.

    Args:
        img_size (int): side length of the image (e.g. 256).
        encoder_stride (int, optional): stride for the encoder. Defaults to 8.
        primary_kernel (int, optional): kernel size of the PrimaryCapsules layer. Defaults to 9.
        primary_stride (int, optional): stride of the PrimaryCapsules layer. Defaults to 2.

    Returns:
        torch.Tensor: tensor of shape (num_primary, 2) of coordinates.
    """
    fmap_size = img_size // encoder_stride
    primary_map_size = (fmap_size - primary_kernel) // primary_stride + 1

    fmap_start = (primary_kernel - 1) / 2
    fmap_coords = torch.arange(primary_map_size) * primary_stride + fmap_start

    img_coords = fmap_coords * encoder_stride

    y_coords, x_coords = torch.meshgrid(img_coords, img_coords, indexing="ij")
    coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)

    num_caps_per_spatial_loc = primary_map_size * primary_map_size
    num_total_primary_caps = 32 * num_caps_per_spatial_loc

    total_coords = coords.repeat_interleave(
        num_total_primary_caps // num_caps_per_spatial_loc
    )

    return total_coords
