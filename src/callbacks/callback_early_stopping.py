from .callback import Callback


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=1, min_delta=0, restore_best_weights=False):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False

    def on_epoch_end(self, epoch, logs=None):
        split = logs.get("phase")
        if split != "val":
            return

        current_loss = logs.get("loss")
        assert (
            current_loss is not None
        ), "No validation loss provided for early stopping criterion."

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in logs["model"].state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state is not None:
                    logs["model"].load_state_dict(self.best_model_state)
                    print(f"Restoring weights to best model at epoch {self.best_epoch}")

                logs["stop"] = self.early_stop
