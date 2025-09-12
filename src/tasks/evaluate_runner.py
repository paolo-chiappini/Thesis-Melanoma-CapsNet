import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from losses import create_combined_loss
from metrics.mig import compute_greedy_mig_score, compute_mig_score, plot_mig_heatmap
from metrics.pairwise_mutual_information import (
    compute_pairwise_mi,
    plot_mi_heatmap,
    summarize_capsule_poses,
)
from utils.evaluation.capsule_activations import compute_capsule_activations

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

        capsule_activations = capsule_activations[:, :-1, :]
        attributes = attributes[:, :-1]

        print(attributes.shape)

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

        greedy_results = compute_greedy_mig_score(
            latent_poses=capsule_activations, true_labels=attributes
        )

        print(greedy_results)

        plot_mig_heatmap(greedy_results, filename="figures/greedy_mig.pdf")
        print(f"Greedy MIG score: {greedy_results['mig_score']}")
