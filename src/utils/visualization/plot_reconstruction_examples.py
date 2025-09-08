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
    max_capsules=8,
    mean=None,
    std=None,
    va_mask_labels=None,
    max_samples=4,
    logger=None,
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
        # Dynamically adjust height: 3 base rows + 1 per capsule
        num_rows = 1 + max_capsules
        fig_height = 3 + max_capsules * 2
        fig, axes = plt.subplots(num_rows, 4, figsize=(12, fig_height))

        # Row 0: original, global, lesion mask
        axes[0, 3].imshow(images[idx].permute(1, 2, 0))
        axes[0, 3].set_title("Original")
        axes[0, 0].imshow(global_recons[idx].permute(1, 2, 0))
        axes[0, 0].set_title("Global Recon")
        axes[0, 1].imshow(lesion_masks[idx][0], cmap="gray")
        axes[0, 1].set_title("Lesion Mask")
        axes[0, 2].imshow((lesion_masks[idx][0] * images[idx]).permute(1, 2, 0))
        axes[0, 2].set_title("Masked image")
        axes[0, 2].axis("off")

        for cap in range(max_capsules):
            row = cap + 1
            axes[row, 0].imshow(capsule_recons[idx, cap].permute(1, 2, 0))
            axes[row, 0].set_title(f"Capsule {cap+1} Recon")

            axes[row, 1].imshow(va_masks[idx, cap], cmap="gray")
            if va_mask_labels and cap < len(va_mask_labels):
                axes[row, 1].set_title(f"VA Mask: {va_mask_labels[cap]}")
            else:
                axes[row, 1].set_title(f"VA Mask {cap+1}")

            masked_image = images[idx] * va_masks[idx, cap].unsqueeze(0)
            axes[row, 2].imshow(masked_image.permute(1, 2, 0))
            axes[row, 2].set_title("Masked VA ROI")

            diff = torch.abs(capsule_recons[idx, cap] - images[idx])
            axes[row, 3].imshow(diff.permute(1, 2, 0))
            axes[row, 3].set_title("Abs Diff")

        # Turn off all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        # Set figure title
        fig.suptitle(
            f"Epoch {epoch} - Sample {idx.item()} - {phase}", fontsize=16, y=1.02
        )

        # Adjust spacing
        plt.subplots_adjust(top=0.95, hspace=0.4)
        plt.tight_layout()

        filename = f"recon_sample_{idx.item()}"

        if logger:
            logger.save_image(fig, filename)
        elif show:
            plt.show()
        else:
            save_path = os.path.join(save_dir, f"{filename}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
