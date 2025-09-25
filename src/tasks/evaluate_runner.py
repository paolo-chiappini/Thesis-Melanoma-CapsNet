import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from tqdm import tqdm

from losses import create_combined_loss
from metrics.attribute_auc import compute_per_attribute_auc, plot_combined_roc_pr_curves
from metrics.mig import compute_mig_score, plot_mig_heatmap
from metrics.pairwise_mutual_information import (
    compute_pairwise_mi,
    plot_mi_heatmap,
    summarize_capsule_poses,
)

from .base_runner import BaseRunner

plt.style.use("science")


class EvaluateRunner(BaseRunner):
    def prepare(self):
        self.prepare_dataset(is_train=False)

        self.build_model(load_weights=True)
        self.loss_criterion = create_combined_loss(
            config=self.config, device=self.device
        )

    def execute(self):
        self.model.eval()

        attribute_names = self.loaders["val"].dataset.dataset.visual_attributes

        attribute_poses_list = []
        attribute_logits_list = []
        malignancy_scores_list = []
        attributes_gt = []
        for batch in tqdm(
            self.loaders["val"], desc="Extracting model outputs", unit="batch"
        ):
            outputs = self.model(batch["images"].to(self.device))
            attribute_poses_list.append(
                outputs["attribute_poses"].detach().cpu().numpy()
            )
            attribute_logits_list.append(
                outputs["attribute_logits"].detach().cpu().numpy()
            )
            malignancy_scores_list.append(
                outputs["malignancy_scores"].detach().cpu().numpy()
            )
            attributes_gt.append(
                batch["visual_attributes_targets"].detach().cpu().numpy()
            )

        attribute_poses = np.concatenate(attribute_poses_list, axis=0)
        attribute_logits = np.concatenate(attribute_logits_list, axis=0)
        malignancy_scores = np.concatenate(malignancy_scores_list, axis=0)
        attributes_gt = np.concatenate(attributes_gt, axis=0)

        summary = summarize_capsule_poses(attribute_poses)
        mi_matrix = compute_pairwise_mi(summary)
        plot_mi_heatmap(mi_matrix, filename="figures/pairwise_mi.pdf")

        print(mi_matrix)

        results = compute_mig_score(
            latent_poses=attribute_poses, true_labels=attributes_gt
        )

        print(results)

        plot_mig_heatmap(results, filename="figures/mig.pdf")
        print(f"MIG score: {results['mig_score']}")

        # greedy_results = compute_greedy_mig_score(
        #     latent_poses=attribute_poses, true_labels=attributes_gt
        # )

        # print(greedy_results)

        # plot_mig_heatmap(greedy_results, filename="figures/greedy_mig.pdf")
        # print(f"Greedy MIG score: {greedy_results['mig_score']}")

        auc_scores = compute_per_attribute_auc(
            y_true=attributes_gt, logits=attribute_logits
        )
        # curves = get_roc_and_pr_curves(
        #     y_true=attributes_gt,
        #     logits=attribute_logits,
        #     attribute_names=self.loaders["val"].dataset.dataset.visual_attributes,
        # )

        # for i, fig in enumerate(curves):
        #     fig.savefig(f"figures/roc_pr_curve_attr_{i}.pdf")

        combined_roc_fig, combined_pr_fig = plot_combined_roc_pr_curves(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )

        roc_save_path = "figures/combined_roc_curves.pdf"
        pr_save_path = "figures/combined_pr_curves.pdf"

        combined_roc_fig.savefig(roc_save_path, dpi=300, bbox_inches="tight")
        combined_pr_fig.savefig(pr_save_path, dpi=300, bbox_inches="tight")

        print(f"Saved ROC at {roc_save_path}")
        print(f"Saved PRC at {pr_save_path}")

        print(f"AUC Scores: {auc_scores}")
        # TODO: continue here
