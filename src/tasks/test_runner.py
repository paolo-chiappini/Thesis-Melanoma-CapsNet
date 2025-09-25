from losses import create_combined_loss
from trainers import get_trainer

from .base_runner import BaseRunner


class TestRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)
        self.compute_weights()

        self.build_model(load_weights=True)
        self.loss_criterion = create_combined_loss(
            config=self.config,
            class_weights=self.weights.get("class_weights"),
            attribute_weights=self.weights.get("attribute_weights"),
            device=self.device,
        )

    def execute(self):
        trainer = get_trainer(
            config=self.config,
            model=self.model,
            data_loader=self.loaders,
            loss_criterion=self.loss_criterion,
            device=self.device,
            checkpoints_dir=self.config["system"]["save_path"],
            save_name=self.config["system"]["save_name"],
        )

        if self.config["system"].get("use_weighted_metrics", False):
            trainer.set_weights(weights_dict=self.weights)

        trainer.test(split="val")
