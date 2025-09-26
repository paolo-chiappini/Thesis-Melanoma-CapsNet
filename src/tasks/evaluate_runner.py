import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
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


def calculate_all_metrics(
    attribute_poses: np.ndarray,
    attribute_logits: np.ndarray,
    attributes_gt: np.ndarray,
    attribute_names: list,
) -> dict:
    """
    Computes all desired scalar metrics from a given set of model outputs and ground truth.

    Args:
        attribute_poses (np.ndarray): the capsule poses (N, K, d).
        attribute_logits (np.ndarray): the predicted attribute logits (N, K).
        attributes_gt (np.ndarray): the ground truth attribute labels (N, K).
        attribute_names (list): A list of strings for the attribute names (K).

    Returns:
        A dictionary where keys are metric names and values are the computed scalar scores.
    """
    metrics = {}

    # MIG scores
    mig_results = compute_mig_score(
        latent_poses=attribute_poses, true_labels=attributes_gt
    )
    metrics["MIG_score"] = mig_results["mig_score"]

    # AUC scores
    auc_scores = compute_per_attribute_auc(
        y_true=attributes_gt, logits=attribute_logits, attribute_names=attribute_names
    )

    auroc_scores = auc_scores.get("auroc", [])
    auprc_scores = auc_scores.get("auprc", [])

    for i, name in enumerate(attribute_names):
        if i < len(auroc_scores):
            metrics[f"AUROC_{name}"] = float(auroc_scores[i])

        if i < len(auprc_scores):
            metrics[f"AUPRC_{name}"] = float(auprc_scores[i])

    if auroc_scores:
        metrics["AUROC_mean"] = np.nanmean(auroc_scores)
    if auprc_scores:
        metrics["AUPRC_mean"] = np.nanmean(auprc_scores)

    return metrics


def bootstrap_evaluate(
    attribute_poses: np.ndarray,
    attribute_logits: np.ndarray,
    attributes_gt: np.ndarray,
    attribute_names: list,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Performs bootstrap resampling to estimate the distribution of evaluation metrics.

    Args:
        attribute_poses (np.ndarray): full set of capsule poses from the validation set (N, K, d).
        attribute_logits (np.ndarray): full set of predicted logits (N, K).
        attributes_gt (np.ndarray): full set of ground truth labels (N, K).
        attribute_names (list): A list of strings for the attribute names (K).
        n_bootstrap (int): number of bootstrap samples to create.

    Returns:
        A pandas DataFrame where each row is the result from one bootstrap sample
        and each column is a metric.
    """
    n_samples = attributes_gt.shape[0]
    bootstrap_results = []

    print(f"Starting bootstrap evaluation with {n_bootstrap} samples...")
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap Iterations", unit="sample"):
        bootstrap_indices = np.random.choice(
            np.arange(n_samples), size=n_samples, replace=True
        )

        poses_sample = attribute_poses[bootstrap_indices]
        logits_sample = attribute_logits[bootstrap_indices]
        gt_sample = attributes_gt[bootstrap_indices]

        metrics = calculate_all_metrics(
            attribute_poses=poses_sample,
            attribute_logits=logits_sample,
            attributes_gt=gt_sample,
            attribute_names=attribute_names,
        )
        bootstrap_results.append(metrics)

    return pd.DataFrame(bootstrap_results)


def plot_metric_comparison(
    summary_df: pd.DataFrame, title: str, metric_prefix: str = "AUC_"
) -> plt.figure:
    """
    Creates a bar chart with error bars for a set of metrics.

    Args:
        summary_df (pd.DataFrame): summary DataFrame from bootstrapping.
        title (str): title for the plot.
        metric_prefix (str): prefix to identify which metrics to plot (e.g. 'AUC_').

    Returns:
        a matplotlib figure containing the bar chart with error bars.
    """
    plot_data = summary_df[summary_df.index.str.startswith(metric_prefix)].copy()

    if plot_data.empty:
        print(
            f"Cannot generate plot '{title}' because no data remains after filtering with prefix '{metric_prefix}'."
        )
        return plt.figure()

    plot_data["Attribute"] = plot_data.index.str.replace(metric_prefix, "", regex=False)

    error = ((plot_data["95% CI upper"] - plot_data["95% CI lower"]) / 2).values

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        data=plot_data,
        x="Mean",
        y="Attribute",
        hue="Attribute",
        palette="viridis",
        ax=ax,
        legend=False,
    )

    ax.errorbar(
        x=plot_data["Mean"],
        y=plot_data["Attribute"],
        xerr=error,
        fmt="none",
        ecolor="black",
        capsize=5,
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Mean Score", fontsize=12)
    ax.set_ylabel("Attribute", fontsize=12)

    if "AUROC" in metric_prefix:
        ax.axvline(x=0.5, color="r", linestyle="--", label="Chance (AUC=0.5)")
        ax.legend()

    ax.grid(alpha=0.4, which="both")
    fig.tight_layout()
    return fig


def plot_distribution(bootstrap_df: pd.DataFrame, metric_name: str, title: str):
    """
    Creates the histogram and KDE of a single metric from the bootstrap results.

    Args:
        bootstrap_df (pd.DataFrame): DataFrame of bootstrapped results.
        metric_name (str): name to identify which metrics to plot.
        title (str): title for the plot.

    Returns:
        a matplotlib figure containing the histogram and KDE of a given metric.
    """
    metric_values = bootstrap_df[metric_name]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(
        metric_values, kde=True, ax=ax, stat="density", label="Bootstrap Distribution"
    )

    ci_lower = metric_values.quantile(0.025)
    ci_upper = metric_values.quantile(0.975)
    mean_val = metric_values.mean()

    ax.axvline(mean_val, color="r", linestyle="-", lw=2, label=f"Mean: {mean_val:.4f}")
    ax.axvspan(
        ci_lower,
        ci_upper,
        color="r",
        alpha=0.1,
        label=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


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

        bootstrap_dataframe = bootstrap_evaluate(
            attribute_poses=attribute_poses,
            attribute_logits=attribute_logits,
            attributes_gt=attributes_gt,
            attribute_names=attribute_names,
            n_bootstrap=1000,
        )

        mean_auroc_dist_fig = plot_distribution(
            bootstrap_dataframe,
            metric_name="AUROC_mean",
            title="Distribution of Mean AUROC from 1000 Bootstrap Samples",
        )
        mean_auprc_dist_fig = plot_distribution(
            bootstrap_dataframe,
            metric_name="AUPRC_mean",
            title="Distribution of Mean AUPRC from 1000 Bootstrap Samples",
        )

        mean_auroc_save_path = "figures/mean_auroc_distribution.pdf"
        mean_auprc_save_path = "figures/mean_auprc_distribution.pdf"

        mean_auroc_dist_fig.savefig(mean_auroc_save_path, dpi=300, bbox_inches="tight")
        mean_auprc_dist_fig.savefig(mean_auprc_save_path, dpi=300, bbox_inches="tight")

        print("> Bootstrap evaluation results:")

        confidence_interval_lower = bootstrap_dataframe.quantile(0.025)
        confidence_interval_upper = bootstrap_dataframe.quantile(0.975)
        mean_scores = bootstrap_dataframe.mean()
        std_devs = bootstrap_dataframe.std()

        summary_dataframe = pd.DataFrame(
            {
                "Mean": mean_scores,
                "Std dev": std_devs,
                "95% CI lower": confidence_interval_lower,
                "95% CI upper": confidence_interval_upper,
            }
        )

        with pd.option_context("display.precision", 4):
            print(summary_dataframe)

        summary_dataframe.to_csv("figures/evaluation_summary_with_confidence.csv")
        print("Saved summary with CI at figures/evaluation_summary_with_confidence.csv")

        auroc_ci_save_path = "figures/auroc_comparison_with_ci.pdf"
        auprc_ci_save_path = "figures/auprc_comparison_with_ci.pdf"

        auroc_comparison_fig = plot_metric_comparison(
            summary_df=summary_dataframe,
            title="Per-attribute AUROC with 95% CIs",
            metric_prefix="AUROC_",
        )
        auroc_comparison_fig.savefig(auroc_ci_save_path, dpi=300, bbox_inches="tight")
        auprc_comparison_fig = plot_metric_comparison(
            summary_df=summary_dataframe,
            title="Per-attribute AUPRC with 95% CIs",
            metric_prefix="AUPRC_",
        )
        auprc_comparison_fig.savefig(auprc_ci_save_path, dpi=300, bbox_inches="tight")

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

        auc_scores = compute_per_attribute_auc(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )

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
