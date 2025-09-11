import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from losses import create_combined_loss
from metrics.mig import compute_mig_score, plot_mig_heatmap
from metrics.pairwise_mutual_information import (
    compute_pairwise_mi,
    plot_mi_heatmap,
    summarize_capsule_poses,
)
from src.utils.evaluation.capsule_activations import compute_capsule_activations

# from utils.visualization import plot_mi_heatmap
from .base_runner import BaseRunner

plt.style.use("science")


class EvaluateRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)

        self.build_model(load_weights=True)
        self.loss_criterion = create_combined_loss(
            config=self.config["trainer"]["loss"]
        )

    def execute(self):
        capsule_activations, attributes = compute_capsule_activations(
            self.model, dataloader=self.loaders["val"], device=self.device
        )

        summary = summarize_capsule_poses(capsule_activations)
        mi_matrix = compute_pairwise_mi(summary)
        plot_mi_heatmap(mi_matrix, filename="figures/pairwise_mi.pdf")

        print(mi_matrix)

        results = compute_mig_score(
            latent_poses=capsule_activations, true_labels=attributes
        )

        print(results)

        plot_mig_heatmap(results, filename="figures/mig.pdf")
        print(f"MIG score: {results['mig_score']}")
