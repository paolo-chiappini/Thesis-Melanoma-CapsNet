from losses import create_combined_loss
from trainers import get_trainer
from utils.evaluation import (
    compute_capsule_activations,
    evaluate_reconstruction,
    mutual_information_capsules,
    summarize_evaluation,
)
from utils.visualization import plot_mi_heatmap

from .base_runner import BaseRunner


class EvaluateRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)

        self.build_model(load_weights=True)
        self.loss_criterion = create_combined_loss(
            config=self.config["trainer"]["loss"]
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

        prepare_batch_func = trainer.prepare_batch

        recon_results = evaluate_reconstruction(
            self.model,
            dataloader=self.loaders["val"],
            device=self.device,
            prepare_batch=prepare_batch_func,
        )
        capsule_activations, attributes = compute_capsule_activations(
            self.model,
            dataloader=self.loaders["val"],
            device=self.device,
            prepare_batch=prepare_batch_func,
        )
        mi_results = mutual_information_capsules(
            capsule_activations, attributes, discrete_attributes=True
        )

        # TODO: check if visual attributes exist in dataset
        attribute_names = self.dataset.visual_attributes
        df_recon, df_mi = summarize_evaluation(
            recon_results, mi_results, attribute_names
        )

        df_recon.to_excel("./plots/reconstructions.xlsx", index=False)
        df_mi.to_excel("./plots/mutual_information.xlsx", index=False)

        plot_mi_heatmap(mi_results, attribute_names=attribute_names)
