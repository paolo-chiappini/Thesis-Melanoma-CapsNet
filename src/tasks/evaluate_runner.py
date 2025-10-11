import math
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
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


def reshape_bootstrap_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Bootstrap"] = df.index

    metric_columns = [
        col
        for col in df.columns
        if col.startswith("AUROC_") or col.startswith("AUPRC_")
    ]

    long_df = df.melt(
        id_vars=["Bootstrap"],
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Value",
    )

    long_df["Prefix"] = long_df["Metric"].str.extract(r"^(AUROC|AUPRC)")
    long_df["Attribute"] = long_df["Metric"].str.extract(r"^[^_]+_(.+)")

    return long_df


def plot_metric_comparison(
    long_df: pd.DataFrame, title: str, metric_prefix: str = "AUC_"
) -> plt.figure:
    """
    Creates a bar chart with error bars for a set of metrics.

    Args:
        long_df (pd.DataFrame): (reshaped) long DataFrame from bootstrapping.
        title (str): title for the plot.
        metric_prefix (str): prefix to identify which metrics to plot (e.g. 'AUC_').

    Returns:
        a matplotlib figure containing the bar chart with error bars.
    """
    plot_data = long_df[long_df["Prefix"] == metric_prefix].copy()

    if plot_data.empty:
        print(f"No data for prefix '{metric_prefix}'")
        return plt.figure()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.boxplot(
        data=plot_data,
        x="Value",
        y="Attribute",
        hue="Attribute",
        palette="viridis",
        ax=ax,
        dodge=False,
        showfliers=True,
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Attribute", fontsize=12)

    if metric_prefix.upper() == "AUROC":
        ax.axvline(x=0.5, color="r", linestyle="--", label="Chance (AUC = 0.5)")
        ax.legend()

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


def plot_normalized_multilabel_cm(
    y_true: np.ndarray,
    logits: np.ndarray,
    attribute_names: list,
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Generates a normalized multilabel confusion matrix.

    The diagonal (i, i) shows the RECALL (True Positive Rate) for attribute i.
    The off-diagonal (i, j) shows the rate of samples that were TRUE for i
    and PREDICTED as j, normalized by the total number of true i samples.
    """
    y_pred = (torch.sigmoid(torch.tensor(logits)) > threshold).int().numpy()
    num_attributes = len(attribute_names)
    conf_matrix = np.zeros((num_attributes, num_attributes), dtype=float)

    for i in range(num_attributes):
        support_i = np.sum(y_true[:, i])

        if support_i == 0:
            continue

        for j in range(num_attributes):
            true_i_and_pred_j = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1))

            conf_matrix[i, j] = true_i_and_pred_j / support_i

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        xticklabels=attribute_names,
        yticklabels=attribute_names,
    )

    ax.set_title(
        "Normalized Multilabel Confusion Matrix (Rows sum to Recall)",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Predicted Attributes", fontsize=12)
    ax.set_ylabel("True Attributes", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_multilabel_metric_heatmap(
    y_true: np.ndarray,
    logits: np.ndarray,
    attribute_names: list,
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Generates a heatmap of key classification metrics (Precision, Recall, F1-Score)
    for each attribute, providing a comprehensive, imbalance-aware summary.
    """
    y_pred = (torch.sigmoid(torch.tensor(logits)) > threshold).int().numpy()

    report = classification_report(
        y_true, y_pred, target_names=attribute_names, output_dict=True, zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    metric_cols = ["precision", "recall", "f1-score", "support"]
    plot_df = report_df.loc[attribute_names, metric_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)

    ax.set_title("Per-Attribute Classification Metrics", fontsize=16, pad=20)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Attributes", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def _plot_single_cm_on_ax(ax, y_true, logits, attribute_name, threshold=0.5):
    """Helper function to plot one confusion matrix on a given matplotlib Axes."""
    y_pred = (torch.sigmoid(torch.tensor(logits)) > threshold).int().numpy()
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        _cm = np.zeros((2, 2), dtype=int)
        _cm[: cm.shape[0], : cm.shape[1]] = cm
        cm = _cm

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot = np.array(
        [
            [f"{count}\n({pct:.1%})" for count, pct in zip(row_val, row_norm)]
            for row_val, row_norm in zip(cm, cm_normalized)
        ]
    )

    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(attribute_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.xaxis.set_ticklabels(["Neg", "Pos"])
    ax.yaxis.set_ticklabels(["Neg", "Pos"])


def plot_combined_confusion_matrices(
    y_true: np.ndarray, logits: np.ndarray, attribute_names: list
) -> plt.Figure:
    """
    Generates a single figure with a grid of confusion matrices, one for each attribute.
    """
    valid_attributes = [
        (k, name)
        for k, name in enumerate(attribute_names)
        if np.unique(y_true[:, k]).size > 1
    ]

    num_plots = len(valid_attributes)
    if num_plots == 0:
        return plt.figure()

    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for i, (k, name) in enumerate(valid_attributes):
        _plot_single_cm_on_ax(
            ax=axes[i], y_true=y_true[:, k], logits=logits[:, k], attribute_name=name
        )

    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Attribute Confusion Matrices", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def plot_logit_distributions(
    y_true: np.ndarray, logits: np.ndarray, attribute_names: List[str]
) -> plt.Figure:
    """
    Generates a faceted KDE plot of logit distributions, stratified by ground truth.

    This plot is a powerful diagnostic tool to visualize classifier calibration and
    discriminative power for each attribute in a multilabel setting.

    Args:
        y_true (np.ndarray): Ground truth labels (N, K), values {0, 1}.
        logits (np.ndarray): Predicted attribute logits (N, K).
        attribute_names (List[str]): List of K attribute names for labeling subplots.

    Returns:
        plt.Figure: A matplotlib Figure object containing the complete faceted plot.
    """
    num_samples, num_attributes = y_true.shape

    data_list = []
    for k, name in enumerate(attribute_names):
        for i in range(num_samples):
            data_list.append(
                {
                    "attribute": name,
                    "logit": logits[i, k],
                    "label": "Present (GT=1)" if y_true[i, k] == 1 else "Absent (GT=0)",
                }
            )
    df = pd.DataFrame(data_list)

    ncols = math.ceil(math.sqrt(num_attributes))
    nrows = math.ceil(num_attributes / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True
    )
    axes = axes.flatten()

    palette = {"Absent (GT=0)": "cornflowerblue", "Present (GT=1)": "tomato"}

    for i, attr_name in enumerate(attribute_names):
        ax = axes[i]
        subset_df = df[df["attribute"] == attr_name]

        sns.kdeplot(
            data=subset_df,
            x="logit",
            hue="label",
            fill=True,
            alpha=0.5,
            palette=palette,
            ax=ax,
            cut=0,
        )

        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            lw=2,
            label="Decision Boundary (logit=0)",
        )

        mean_absent = subset_df[subset_df["label"] == "Absent (GT=0)"]["logit"].mean()
        mean_present = subset_df[subset_df["label"] == "Present (GT=1)"]["logit"].mean()

        ax.axvline(x=mean_absent, color=palette["Absent (GT=0)"], linestyle=":", lw=2)
        ax.axvline(x=mean_present, color=palette["Present (GT=1)"], linestyle=":", lw=2)

        ax.set_title(attr_name, fontsize=12)
        ax.set_xlabel("Logit Value")
        ax.set_ylabel("Density" if i % ncols == 0 else "")

        if i == 0:
            ax.legend()
        else:
            ax.get_legend().remove()

    for j in range(num_attributes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution of Predicted Logits, Stratified by Ground-Truth Label",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()

    return fig


def _plot_embedding_on_ax(ax, projected_embeddings, labels, title):
    """Helper function to create a styled scatter plot on a given Axes object."""
    df = pd.DataFrame(
        {
            "x": projected_embeddings[:, 0],
            "y": projected_embeddings[:, 1],
            "label": ["Present (GT=1)" if l == 1 else "Absent (GT=0)" for l in labels],
        }
    )

    palette = {"Absent (GT=0)": "cornflowerblue", "Present (GT=1)": "tomato"}

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=palette,
        ax=ax,
        alpha=0.7,
        s=20,  # Adjust point size
    )
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)


def generate_embedding_visualizations(
    attribute_poses: np.ndarray,
    va_labels: np.ndarray,
    attribute_names: List[str],
    method: Literal["umap", "tsne"] = "umap",
) -> plt.Figure:
    """
    Generates a faceted plot of 2D embeddings for each attribute's poses,
    colored by ground-truth labels.

    Args:
        attribute_poses (np.ndarray): High-dimensional poses (N, K, pose_dim).
        va_labels (np.ndarray): Ground truth labels (N, K).
        attribute_names (List[str]): List of attribute names for subplot titles.
        method (str): The reduction method to use, 'umap' or 'tsne'.

    Returns:
        plt.Figure: A matplotlib Figure containing the faceted plot.
    """
    if method == "umap" and umap is None:
        print("UMAP not found, switching to t-SNE.")
        method = "tsne"

    num_attributes = len(attribute_names)
    ncols = math.ceil(math.sqrt(num_attributes))
    nrows = math.ceil(num_attributes / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten()

    print(f"Generating {method.upper()} plots for {num_attributes} attributes...")
    for k, name in enumerate(attribute_names):
        ax = axes[k]
        poses_k = attribute_poses[:, k, :]
        labels_k = va_labels[:, k]

        if method == "umap":
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            )
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)

        projected_embeddings = reducer.fit_transform(poses_k)

        _plot_embedding_on_ax(ax, projected_embeddings, labels_k, name)

    for j in range(num_attributes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{method.upper()} Visualization of Pose Embeddings per Attribute",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()

    return fig


def plot_pose_centroids(
    attribute_poses: np.ndarray,
    va_labels: np.ndarray,
    attribute_names: List[str],
    method: Literal["umap", "tsne"] = "umap",
) -> plt.Figure:
    """
    Visualizes the mean centroids of 'Present' vs 'Absent' poses for all
    attributes in a single, shared embedding space.
    """
    N, K, P = attribute_poses.shape

    all_poses_flat = attribute_poses.reshape(N * K, P)

    if method == "umap" and umap:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)

    projected_flat = reducer.fit_transform(all_poses_flat)
    projected_poses = projected_flat.reshape(N, K, 2)

    centroids = []
    for k, name in enumerate(attribute_names):
        poses_k_2d = projected_poses[:, k, :]
        labels_k = va_labels[:, k]

        absent_centroid = poses_k_2d[labels_k == 0].mean(axis=0)
        present_centroid = poses_k_2d[labels_k == 1].mean(axis=0)

        centroids.append(
            {
                "attribute": name,
                "label": "Absent (GT=0)",
                "x": absent_centroid[0],
                "y": absent_centroid[1],
            }
        )
        centroids.append(
            {
                "attribute": name,
                "label": "Present (GT=1)",
                "x": present_centroid[0],
                "y": present_centroid[1],
            }
        )

    centroid_df = pd.DataFrame(centroids)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {"Absent (GT=0)": "blue", "Present (GT=1)": "red"}

    sns.scatterplot(
        data=centroid_df,
        x="x",
        y="y",
        style="label",
        hue="label",
        palette=palette,
        s=100,
        ax=ax,
    )

    for i, row in centroid_df.iterrows():
        ax.text(row["x"] + 0.05, row["y"], row["attribute"], fontsize=9)

    ax.set_title(f"{method.upper()} of Pose Centroids", fontsize=16)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Class Centroid")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig


def generate_routing_visualization(
    model,
    dataloader,
    image_indices,
    attribute_names,
    device,
    save_path="routing_analysis.pdf",
):
    """
    Generates and saves the "Routing Adjacency Matrix" visualization for
    specified images.

    Args:
        model (nn.Module): Your trained Capsule Network model.
        dataloader (DataLoader): The dataloader for the validation or test set.
        image_indices (list[int]): A list of integer indices of the images to visualize.
        attribute_names (list[str]): A list of names for the attribute capsules (Y-axis labels).
        device (torch.device): The device to run the model on.
        save_path (str): Path to save the output PDF file.
    """
    model.to(device)
    model.eval()

    num_images = len(image_indices)
    fig, axes = plt.subplots(
        num_images,
        2,
        figsize=(15, 5 * num_images),
        gridspec_kw={"width_ratios": [1, 3]},  # Make the heatmap wider
    )
    fig.suptitle("Analysis of Routing Adjacency Matrices", fontsize=20, y=1.02)

    dataset = dataloader.dataset

    with torch.no_grad():
        for i, img_idx in enumerate(image_indices):
            # 1. Get the specific image and label from the dataset
            data = dataset[img_idx]
            image = data["images"]
            label = data["visual_attributes_targets"]
            image = image.unsqueeze(0).to(
                device
            )  # Add batch dimension and move to device
            label = label.unsqueeze(0)

            # 2. Perform a forward pass
            outputs = model(image)  # y_labels=None for eval mode

            # 3. Extract the coupling coefficients
            # Shape: (1, K, P, 1) -> squeeze to (K, P)
            coupling_coeffs = outputs["coupling_coefficients"].squeeze().cpu().numpy()

            # --- Plotting ---

            # Get the original image for display (un-normalize if necessary)
            # This depends on your dataset implementation. We'll just show the tensor.
            img_to_show = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            # If normalized, you'd un-normalize here for correct colors
            # img_to_show = img_to_show * std + mean

            ax_img = axes[i, 0]
            ax_heatmap = axes[i, 1]

            # Plot the original image
            ax_img.imshow(np.clip(img_to_show, 0, 1))
            ax_img.axis("off")

            # Find which attributes are present
            present_attrs_indices = torch.where(label.squeeze() == 1)[0].tolist()
            present_attrs_names = [attribute_names[j] for j in present_attrs_indices]
            title_text = f"Image #{img_idx}\nGround Truth: {', '.join(present_attrs_names) or 'None'}"
            ax_img.set_title(title_text, fontsize=12)

            # Plot the heatmap
            sns.heatmap(
                coupling_coeffs,
                ax=ax_heatmap,
                cmap="viridis",
                cbar=True,
                cbar_kws={"label": "Coupling Coefficient Value"},
            )
            ax_heatmap.set_title("Routing Adjacency Matrix", fontsize=12)
            ax_heatmap.set_ylabel("Attribute Capsule Index")
            ax_heatmap.set_yticks(np.arange(len(attribute_names)) + 0.5)
            ax_heatmap.set_yticklabels(attribute_names, rotation=0)
            ax_heatmap.set_xlabel("Primary Capsule Index")

    plt.tight_layout(rect=[0, 0, 1, 1])
    print(f"Saving visualization to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


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

        generate_routing_visualization(
            model=self.model,
            dataloader=self.loaders["val"],
            image_indices=[0, 1, 2],
            attribute_names=attribute_names,
            device=self.device,
        )

        plt.style.use("science")

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

        logit_dist_fig = plot_logit_distributions(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )

        logit_dist_save_path = "figures/logit_distributions.pdf"
        logit_dist_fig.savefig(logit_dist_save_path, dpi=300, bbox_inches="tight")
        print(f"Saved logit distribution plot to: {logit_dist_save_path}")

        multilabel_cm_fig = plot_normalized_multilabel_cm(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )
        multilabel_cm_fig.savefig(
            "figures/multilabel_confusion_matrix.pdf", dpi=300, bbox_inches="tight"
        )
        print(
            "Saved multilabel confusion matrix to: figures/multilabel_confusion_matrix.pdf"
        )

        cm_grid_fig = plot_combined_confusion_matrices(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )
        cm_grid_fig.savefig(
            "figures/combined_confusion_matrices.pdf", dpi=300, bbox_inches="tight"
        )
        print(
            "Saved grid of confusion matrices to: figures/combined_confusion_matrices.pdf"
        )

        multilabel_metric_fig = plot_multilabel_metric_heatmap(
            y_true=attributes_gt,
            logits=attribute_logits,
            attribute_names=attribute_names,
        )
        multilabel_metric_fig.savefig(
            "figures/multilabel_metric_heatmap.pdf", dpi=300, bbox_inches="tight"
        )
        print("Saved metrics heatmap to: figures/multilabel_metric_heatmap.pdf")

        bootstrap_dataframe = bootstrap_evaluate(
            attribute_poses=attribute_poses,
            attribute_logits=attribute_logits,
            attributes_gt=attributes_gt,
            attribute_names=attribute_names,
            n_bootstrap=1000,
        )

        bootstrap_dataframe.to_csv("figures/bootstrap_dataframe.csv")
        print("Saved summary with CI at figures/bootstrap_dataframe.csv")

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

        summary_dataframe.to_csv("figures/evaluation_summary_with_confidence.csv")
        print("Saved summary with CI at figures/evaluation_summary_with_confidence.csv")

        auroc_ci_save_path = "figures/auroc_comparison_with_ci.pdf"
        auprc_ci_save_path = "figures/auprc_comparison_with_ci.pdf"

        long_bootstrap_df = reshape_bootstrap_df(df=bootstrap_dataframe)

        auroc_comparison_fig = plot_metric_comparison(
            long_df=long_bootstrap_df,
            title="Per-attribute AUROC with 95% CIs",
            metric_prefix="AUROC",
        )
        auroc_comparison_fig.savefig(auroc_ci_save_path, dpi=300, bbox_inches="tight")
        auprc_comparison_fig = plot_metric_comparison(
            long_df=long_bootstrap_df,
            title="Per-attribute AUPRC with 95% CIs",
            metric_prefix="AUPRC",
        )
        auprc_comparison_fig.savefig(auprc_ci_save_path, dpi=300, bbox_inches="tight")

        summary = summarize_capsule_poses(attribute_poses)
        mi_matrix = compute_pairwise_mi(summary)
        plot_mi_heatmap(mi_matrix, filename="figures/pairwise_mi.pdf")

        # print(mi_matrix)

        results = compute_mig_score(
            latent_poses=attribute_poses, true_labels=attributes_gt
        )

        # print(results)

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

        print("\nGenerating embedding visualizations...")

        umap_faceted_fig = generate_embedding_visualizations(
            attribute_poses=attribute_poses,
            va_labels=attributes_gt,
            attribute_names=attribute_names,
            method="umap",
        )
        umap_faceted_fig.savefig("figures/umap_faceted_embeddings.pdf", dpi=300)
        print("Saved faceted UMAP plot to figures/umap_faceted_embeddings.pdf")

        centroid_fig = plot_pose_centroids(
            attribute_poses=attribute_poses,
            va_labels=attributes_gt,
            attribute_names=attribute_names,
            method="umap",
        )
        centroid_fig.savefig("figures/umap_pose_centroids.pdf", dpi=300)
        print("Saved UMAP centroid plot to figures/umap_pose_centroids.pdf")
