import os
from .callback import Callback
import matplotlib.pyplot as plt


class ReconstructionCallback(Callback):
    """
    Callback to visualize the reconstruction of images.
    """

    def __init__(self, frequency=5, show=False, save_dir="reconstructions"):
        super().__init__()
        self.frequency = frequency
        self.show = show
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def on_reconstruction(self, images, reconstructions, epoch, phase):
        if phase == "test" and epoch % self.frequency == 0:
            print(f"Reconstruction at epoch {epoch} - {phase}")
            images = images.cpu().detach()
            reconstructions = reconstructions.cpu().detach()
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
