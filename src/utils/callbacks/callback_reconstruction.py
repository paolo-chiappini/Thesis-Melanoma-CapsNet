import os
import torch
from .callback import Callback
import matplotlib.pyplot as plt


class ReconstructionCallback(Callback):
    """
    Callback to visualize the reconstruction of images.
    """

    def __init__(
        self, frequency=5, show=False, save_dir="reconstructions", mean=None, std=None
    ):
        super().__init__()
        self.frequency = frequency
        self.show = show
        self.save_dir = save_dir
        self.mean = mean
        self.std = std

        os.makedirs(self.save_dir, exist_ok=True)

    def on_reconstruction(self, images, reconstructions, epoch, phase):
        if phase == "val" and epoch % self.frequency == 0:
            print(f"Reconstruction at epoch {epoch} - {phase}")
            images = images.cpu().detach()
            reconstructions = reconstructions.cpu().detach()

            # denormalization for plots
            if self.mean is not None and self.std is not None:
                if not isinstance(self.mean, torch.Tensor):
                    self.mean = torch.tensor(self.mean)
                if not isinstance(self.std, torch.Tensor):
                    self.std = torch.tensor(self.std)

                self.mean = self.mean.to(images.device).view(1, -1, 1, 1)
                self.std = self.std.to(images.device).view(1, -1, 1, 1)

                images = images * self.std + self.mean
                reconstructions = reconstructions * self.std + self.mean
            self._plot_reconstruction(images, reconstructions, epoch, phase)

    def _plot_reconstruction(self, images, reconstructions, epoch, phase):
        plt.rcParams.update({"font.size": 8})
        fig, axes = plt.subplots(2, len(images), figsize=(12, 4))
        for i in range(len(images)):
            axes[0, i].imshow(images[i].permute(1, 2, 0))
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0))
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.suptitle(f"Epoch {epoch} - {phase.capitalize()} Reconstructions")
        if self.show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch}.png"))
            plt.close()
