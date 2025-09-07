import os

import matplotlib.pyplot as plt
import torch


def denormalize(x, mean=None, std=None):
    if mean is not None and std is not None:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        mean = mean.view(1, -1, 1, 1).to(x.device)
        std = std.view(1, -1, 1, 1).to(x.device)
        return x * std + mean
    return x


def filter_nonzero_va_masks(va_masks):
    # Keep only examples where at least one va_mask is not all zero
    mask_presence = va_masks.view(va_masks.size(0), va_masks.size(1), -1).sum(dim=2)
    valid_indices = (mask_presence > 0).any(dim=1).nonzero(as_tuple=True)[0]
    return valid_indices


def plot_reconstruction_examples(
    images,
    global_recons,
    capsule_recons,
    lesion_masks,
    va_masks,
    epoch,
    phase,
    save_dir="reconstructions",
    show=False,
    max_capsules=7,
    mean=None,
    std=None,
    va_mask_labels=None,
    max_samples=4,
):
    os.makedirs(save_dir, exist_ok=True)

    images = images.cpu().detach()
    global_recons = global_recons.cpu().detach()
    capsule_recons = capsule_recons.cpu().detach()
    lesion_masks = lesion_masks.cpu().detach()
    va_masks = va_masks.cpu().detach()

    images = denormalize(images, mean, std)
    global_recons = denormalize(global_recons, mean, std)
    capsule_recons = denormalize(capsule_recons, mean, std)

    valid_indices = filter_nonzero_va_masks(va_masks)
    if len(valid_indices) == 0:
        print("[plot_reconstruction_examples] No non-zero VA masks found in batch.")
        return

    num_samples = min(max_samples, len(valid_indices))

    for idx in valid_indices[:num_samples]:
        fig, axes = plt.subplots(
            3 + max_capsules, 4, figsize=(12, 3 + 3 * max_capsules)
        )

        # Row 0: original, global, lesion mask
        axes[0, 0].imshow(images[idx].permute(1, 2, 0))
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(global_recons[idx].permute(1, 2, 0))
        axes[0, 1].set_title("Global Recon")
        axes[0, 2].imshow(lesion_masks[idx][0], cmap="gray")
        axes[0, 2].set_title("Lesion Mask")
        axes[0, 3].imshow((lesion_masks[idx][0] * images[idx]).permute(1, 2, 0))
        axes[0, 3].set_title("Masked image")
        axes[0, 3].axis("off")

        for cap in range(max_capsules):
            # Capsule recon
            axes[cap + 1, 0].imshow(capsule_recons[idx, cap].permute(1, 2, 0))
            axes[cap + 1, 0].set_title(f"Capsule {cap+1} Recon")

            # VA mask
            axes[cap + 1, 1].imshow(va_masks[idx, cap], cmap="gray")
            if va_mask_labels and cap < len(va_mask_labels):
                label = va_mask_labels[cap]
                axes[cap + 1, 1].set_title(f"VA Mask: {label}")
            else:
                axes[cap + 1, 1].set_title(f"VA Mask {cap+1}")

            # Absolute difference
            diff = torch.abs(capsule_recons[idx, cap] - images[idx])
            axes[cap + 1, 2].imshow(diff.permute(1, 2, 0))
            axes[cap + 1, 2].set_title("Abs Diff")

            axes[cap + 1, 3].axis("off")

            masked_image = images[idx] * va_masks[idx, cap].unsqueeze(0)
            axes[cap + 1, 3].imshow(masked_image.permute(1, 2, 0))
            axes[cap + 1, 3].set_title("Image * Mask")

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        plt.suptitle(f"Epoch {epoch} - Sample {idx.item()} - {phase}")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"recon_sample_{idx.item()}.png")
        plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
