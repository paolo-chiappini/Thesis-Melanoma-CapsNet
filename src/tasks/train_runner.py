import torch

from callbacks import CallbackManager, get_callbacks
from losses import create_combined_loss
from trainers import get_trainer
from utils.logs.tb_logger import TBLogger

from .base_runner import BaseRunner


class TrainRunner(BaseRunner):
    def __init__(self, config, model_path=None, cpu_override=False, **kwargs):
        super().__init__(config, model_path, cpu_override, **kwargs)
        self.no_save = kwargs.get("no_save")

    def prepare(self):
        self.prepare_dataset(is_train=True)
        self.compute_weights()
        # self.update_sampler() # Damages training beyond measure

        self.build_model(load_weights=False)
        self.loss_criterion = create_combined_loss(
            config=self.config,
            class_weights=self.weights.get("class_weights"),
            attribute_weights=self.weights.get("attribute_weights"),
            device=self.device,
        )

    # TODO: remove or fix
    def update_sampler(self):
        class_weights = self.weights.get("class_weights", None)
        if class_weights is not None:
            original_dataset = self.loaders["train"].dataset

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(original_dataset),
                replacement=True,
            )

            # recreate data loader with sampler
            self.loaders["train"] = torch.utils.data.DataLoader(
                dataset=original_dataset,
                batch_size=self.loaders["train"].batch_size,
                sampler=sampler,
                num_workers=self.loaders["train"].num_workers,
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

        # logger = None if self.no_save else TrainingLogger(config=self.config)
        logger = None if self.no_save else TBLogger(config=self.config)

        callbacks = get_callbacks(callback_config=self.config.get("callbacks", []))
        callback_manager = CallbackManager(callbacks=callbacks, logger=logger)

        print("Running training with class weights", self.weights.get("class_weights"))

        trainer.set_logger(logger=logger)

        trainer.run(self.config["trainer"]["epochs"], callback_manager=callback_manager)
        trainer.test(split="val")

        if logger is not None:
            logger.close()
