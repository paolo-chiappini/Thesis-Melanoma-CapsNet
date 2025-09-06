import os

import matplotlib.pyplot as plt
import torch

from .callback import Callback


class CurrentEpochExampleCallback(Callback):
    """
    Callback to visualize reconstructions during training, including:
    - Global reconstruction
    - Per-capsule reconstructions aligned with va_masks
    """

    def __init__(
        self,
        frequency=1,
        track="train",
        show=False,
        save_dir="reconstructions",
        mean=None,
        std=None,
        max_capsules=7,
    ):
        super().__init__()
        self.frequency = frequency
        self.track = track
        self.show = show
        self.save_dir = save_dir
        self.mean = mean
        self.std = std
        self.max_capsules = max_capsules

        os.makedirs(self.save_dir, exist_ok=True)

    def _denormalize(self, x):
        if self.mean is not None and self.std is not None:
            if not isinstance(self.mean, torch.Tensor):
                self.mean = torch.tensor(self.mean)
            if not isinstance(self.std, torch.Tensor):
                self.std = torch.tensor(self.std)

            self.mean = self.mean.view(1, -1, 1, 1).to(x.device)
            self.std = self.std.view(1, -1, 1, 1).to(x.device)
            return x * self.std + self.mean
        return x

    # TODO: this is not a valid extension. Temporarily used in the MSR trainer.
    def on_reconstruction(
        self,
        images,
        global_reconstructions,
        capsule_reconstructions,
        lesion_masks,
        va_masks,
        epoch,
        phase,
    ):
        if phase != self.track or epoch % self.frequency != 0:
            return

        print(f"[CurrentEpochExampleCallback] Epoch {epoch} - Phase: {phase}")
        images = images.cpu().detach()
        global_reconstructions = global_reconstructions.cpu().detach()
        capsule_reconstructions = capsule_reconstructions.cpu().detach()
        lesion_masks = lesion_masks.cpu().detach()
        va_masks = va_masks.cpu().detach()

        # # denormalize
        # images = self._denormalize(images)
        # global_reconstructions = self._denormalize(global_reconstructions)
        # capsule_reconstructions = self._denormalize(capsule_reconstructions)

        self._plot_all(
            images,
            global_reconstructions,
            capsule_reconstructions,
            lesion_masks,
            va_masks,
            epoch,
            phase,
        )

    def _plot_all(
        self,
        images,
        global_recons,
        capsule_recons,
        lesion_masks,
        va_masks,
        epoch,
        phase,
    ):
        # batch_size = min(4, images.shape[0])  # Show up to 4 samples
        batch_size = 4
        for i in range(batch_size):
            fig, axes = plt.subplots(
                3 + self.max_capsules, 4, figsize=(12, 3 + 3 * self.max_capsules)
            )

            # Row 0: Original image and global info
            axes[0, 0].imshow(images[i].permute(1, 2, 0))
            axes[0, 0].set_title("Original")
            axes[0, 1].imshow(global_recons[i].permute(1, 2, 0))
            axes[0, 1].set_title("Global Recon")
            axes[0, 2].imshow(lesion_masks[i][0], cmap="gray")
            axes[0, 2].set_title("Lesion Mask")
            axes[0, 3].axis("off")

            # Row 1 to N: Per-capsule recons and va_masks
            for cap in range(self.max_capsules):
                axes[cap + 1, 0].imshow(capsule_recons[i, cap].permute(1, 2, 0))
                axes[cap + 1, 0].set_title(f"Capsule {cap+1} Recon")
                axes[cap + 1, 1].imshow(va_masks[i, cap], cmap="gray")
                axes[cap + 1, 1].set_title(f"VA Mask {cap+1}")

                # Difference map
                diff = torch.abs(capsule_recons[i, cap] - images[i])
                axes[cap + 1, 2].imshow(diff.permute(1, 2, 0))
                axes[cap + 1, 2].set_title("Abs Diff")

                axes[cap + 1, 3].axis("off")

            for ax_row in axes:
                for ax in ax_row:
                    ax.axis("off")

            plt.suptitle(f"Epoch {epoch} - Sample {i} - {phase}")
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"example_recon_sample_{i}.png")
            plt.savefig(save_path)
            plt.close()
