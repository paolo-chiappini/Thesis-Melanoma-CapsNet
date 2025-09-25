from typing import Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


def masks_to_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Compute bbox (xmin, ymin, xmax, ymax) for a single mask tensor [H, W] binary.
    If the mask is empty, it return the whole image as (0, 0, W, H).

    Args:
        mask (torch.Tensor): segmentation mask for the lesion or attribute.

    Returns:
        Tuple[int, int, int, int]: (xmin, ymin, xmax, ymax) bbox coordinates.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    _, H, W = mask.shape
    ys = torch.any(mask, dim=1).nonzero(as_tuple=False)
    xs = torch.any(mask, dim=0).nonzero(as_tuple=False)

    if ys.numel() == 0 or xs.numel() == 0:
        return 0, 0, W, H

    ymin = int(ys.min().item())
    ymax = int(ys.max().item()) + 1
    xmin = int(xs.min().item())
    xmax = int(xs.max().item()) + 1

    return xmin, ymin, xmax, ymax


def pad_bbox(
    bbox: Tuple[int, int, int, int], pad: int, H: int, W: int
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox

    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = max(W, xmax + pad)
    ymax = max(H, ymax + pad)

    return xmin, ymin, xmax, ymax


def crop_tensor_to_bbox(
    tensor: torch.Tensor, bbox: Tuple[int, int, int, int]
) -> torch.Tensor:
    xmin, ymin, xmax, ymax = bbox

    if tensor.dim() == 3:
        return tensor[:, ymin:ymax, xmin:xmax]
    else:
        raise ValueError(f"Unsupported tensor shape for crop, got {tensor.shape}")


def fill_background_with_avg(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Replace black background with avergae color inside the mask for better stability of LPIPS.

    Args:
        image (torch.Tensor): original ground truth image (3, H, W).
        mask (torch.Tensor): binary segmentation mask (1, H, W).

    Returns:
        torch.Tensor: the image with the background set to the average inside the masked region.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    masked = image * mask
    mask_size = mask.sum().clamp(min=1.0)
    avg = masked.sum(dim=(1, 2)) / mask_size
    background = avg.view(-1, 1, 1).expand_as(image)
    output = image * mask + background * (1 - mask)
    return output


class MSRPerceptualLossLPIPS(nn.Module):
    def __init__(self, config: dict, device: torch.device, **kwargs: dict):
        super(MSRPerceptualLossLPIPS, self).__init__()
        self.global_config = config
        self.lpips_config = config["trainer"]["loss"]["components"][
            "MSRPerceptualLossLPIPS"
        ]
        self.device = device

        self.lpips_net = lpips.LPIPS(net=self.lpips_config.get("lpips_net", "vgg")).to(
            self.device
        )
        self.lpips_size = self.lpips_config.get("lpips_size", 224)

    def _prepare_pair_for_lpips(
        self, target: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare single input pair for LPIPS.
        - If mask has few pixels, return None to use the L2 fallback.
        - Otherwise, crop bbox (with padding), resize and normalize to [-1, 1].

        Args:
            target (torch.Tensor): target image (ground truth).
            reconstruction (torch.Tensor): reconstructed image.
            mask (torch.Tensor): segmentation mask.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: returns tensors of shape (3, lpips_size, lpips_size).
        """
        H = target.shape[1]
        W = target.shape[2]

        num_pixels = int(mask.sum().item())
        if num_pixels < self.lpips_config.get("min_mask_pixels_for_crop", 64):
            # image too small/sparse for good performance on LPIPS
            return None

        bbox = masks_to_bbox(mask=mask)
        bbox = pad_bbox(bbox=bbox, pad=self.lpips_config.get("bbox_pad", 8), H=H, W=W)
        target_crop = crop_tensor_to_bbox(target, bbox)
        reconstruction_crop = crop_tensor_to_bbox(reconstruction, bbox)

        # resize to lpips_size
        target_crop = F.interpolate(
            target_crop.unsqueeze(0),
            size=(self.lpips_size, self.lpips_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        reconstruction_crop = F.interpolate(
            reconstruction_crop.unsqueeze(0),
            size=(self.lpips_size, self.lpips_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # normalize in [-1, 1]
        target_crop = target_crop * 2.0 - 1.0
        reconstruction_crop = reconstruction_crop * 2.0 - 1.0

        return target_crop, reconstruction_crop

    def forward(
        self, targets: torch.Tensor, reconstructions: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked LPIPS over a batch of data.
        """
        B = targets.shape[0]
        device = targets.device

        lpips_input_targets = []
        lpips_input_reconstructions = []
        mask_fallback_indices = []  # indices of masks that are too small

        for i in range(B):
            target = targets[i].detach().cpu()
            reconstruction = reconstructions[i].detach().cpu()
            mask = masks[i].detach().cpu()

            prepared = self._prepare_pair_for_lpips(target, reconstruction, mask)
            if prepared is None:
                mask_fallback_indices.append(i)
                continue

            t_crop, r_crop = prepared
            lpips_input_targets.append(t_crop.unsqueeze(0))
            lpips_input_reconstructions.append(r_crop.unsqueeze(0))

        loss_total = torch.tensor(0.0, device=device)
        n_terms = 0

        # apply LPIPS
        if len(lpips_input_targets) > 0 and self.lpips_net is not None:
            batch_targets = torch.cat(lpips_input_targets, dim=0).to(self.device)
            batch_reconstructions = torch.cat(lpips_input_reconstructions, dim=0).to(
                self.device
            )

            with torch.no_grad():
                lpips_vals = self.lpips_net(batch_targets, batch_reconstructions)
            lpips_vals = lpips_vals.view(lpips_vals.shape[0], -1).mean(dim=1)
            loss_total = loss_total + lpips_vals.sum().to(device)
            n_terms += lpips_vals.shape[0]

        # fallback to MSE
        if len(mask_fallback_indices) > 0:
            mse_sum = 0.0
            count = 0
            for i in mask_fallback_indices:
                m = masks[i].to(device)
                t = targets[i].to(device)
                r = reconstructions[i].to(device)

                area = m.sum().clamp(min=1.0)
                diff = (t - r) ** 2
                masked_err = (diff * m.unsqueeze(0)).sum() / area
                mse_sum += masked_err

                count += 1
            loss_total += mse_sum
            n_terms += count

        if n_terms == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss_total / float(n_terms)
