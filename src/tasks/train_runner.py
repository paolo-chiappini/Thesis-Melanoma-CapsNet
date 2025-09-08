from callbacks import CallbackManager, get_callbacks
from losses import create_combined_loss
from trainers import get_trainer
from utils.logs.training_logger import TrainingLogger

from .base_runner import BaseRunner


class TrainRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=True)
        self.compute_weights()

        self.build_model(load_weights=False)
        self.loss_criterion = create_combined_loss(
            config=self.config["trainer"]["loss"],
            class_weights=self.weights.get("class_weights"),
            attribute_weights=self.weights.get("attribute_weights"),
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

        logger = TrainingLogger(config=self.config)

        callbacks = get_callbacks(callback_config=self.config.get("callbacks", []))
        callback_manager = CallbackManager(callbacks=callbacks, logger=logger)

        print("Running training with class weights", self.weights.get("class_weights"))

        trainer.set_logger(logger=logger)

        trainer.run(self.config["trainer"]["epochs"], callback_manager=callback_manager)
        trainer.test(split="val")
