from .trainer_base import BaseTrainer


class AutoEncoderTrainer(BaseTrainer):

    def prepare_batch(self, batch):
        images, _, _, _ = batch
        images = images.to(self.device)
        return {"inputs": (images,), "targets": images}

    def compute_loss(self, outputs, batch_data):
        return self.criterion(outputs, batch_data["targets"])

    def compute_metrics(self, outputs, batch_data):
        return 0.0
