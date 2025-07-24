import os
from .callback import Callback
import matplotlib.pyplot as plt


class PlotCallback(Callback):
    """
    Callback to plot the training and validation loss and accuracy.
    """

    def __init__(self, show=False, save_dir="plots", filename="loss_accuracy.png"):
        super().__init__()
        self.show = show
        self.save_dir = save_dir
        self.filename = filename

        os.makedirs(self.save_dir, exist_ok=True)

        self.epoch_loss = {"train": [], "val": []}
        self.epoch_accuracy = {"train": [], "val": []}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            phase = logs.get("phase")
            self.epoch_loss[phase].append(logs.get("loss"))
            self.epoch_accuracy[phase].append(logs.get("accuracy"))

            if phase == "val":
                self._plot(epoch=epoch)

    def _plot(self, epoch):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch_loss["train"], label="Train Loss")
        plt.plot(self.epoch_loss["val"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_accuracy["train"], label="Train Accuracy")
        plt.plot(self.epoch_accuracy["val"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.suptitle(f"Epoch {epoch} - Training and Validation Metrics")

        if self.show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_dir, self.filename))
            plt.close()
