from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_per_attribute_auc(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """
    Computes per-attribute AUROC and AUPRC.

    Args:
        y_true (np.ndarray): Ground truth labels (N, K), values {0, 1}.
        logits (np.ndarray): Predicted attribute logits (N, K).

    Returns:
        dict: {
            'auroc: List of AUROC values per each attribute (K, ),
            'auprc: List of AUPRC values per each attribute (K, )
        }
    """
    prob_np = torch.sigmoid(torch.tensor(logits)).detach().cpu().numpy()

    num_attributes = y_true.shape[1]
    auroc_list = []
    auprc_list = []

    for k in range(num_attributes):
        y_true_k = y_true[:, k]
        prob_k = prob_np[:, k]

        if np.unique(y_true_k).size == 1:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = roc_auc_score(y_true_k, prob_k)
            auprc = average_precision_score(y_true_k, prob_k)

        auroc_list.append(auroc)
        auprc_list.append(auprc)

    return {"auroc": auroc_list, "auprc": auprc_list}


def get_roc_and_pr_curves(
    y_true: np.ndarray,
    logits: np.ndarray,
    attribute_names: Optional[List[str]] = None,
) -> List[plt.Figure]:
    """
    Generate ROC and Precision-Recall curve figures for each attribute.

    Args:
        y_true (np.ndarray): Ground truth labels (N, K), values {0, 1}.
        logits (np.ndarray): Predicted attribute logits (N, K).
        attribute_names (List[str], optional): List of attribute names for labeling.

    Returns:
        List[plt.Figure]: List of matplotlib Figure objects (1 per attribute)
    """
    prob_np = torch.sigmoid(torch.tensor(logits)).detach().cpu().numpy()

    num_attributes = y_true.shape[1]
    figures = []

    for k in range(num_attributes):
        y_true_k = y_true[:, k]
        prob_k = prob_np[:, k]

        if np.unique(y_true_k).size < 2:
            print(f"Skipping attribute {k} (only one class present)")
            continue

        fpr, tpr, _ = roc_curve(y_true_k, prob_k)
        precision, recall, _ = precision_recall_curve(y_true_k, prob_k)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        attr_name = attribute_names[k] if attribute_names else f"Attribute {k}"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        axs[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_title(f"ROC — {attr_name}")
        axs[0].legend()

        axs[1].plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
        axs[1].set_xlabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[1].set_title(f"Precision-Recall — {attr_name}")
        axs[1].legend()

        fig.suptitle(f"Metrics for {attr_name}", fontsize=14)
        fig.tight_layout()
        figures.append(fig)

    return figures


def plot_combined_roc_pr_curves(
    y_true: np.ndarray,
    logits: np.ndarray,
    attribute_names: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generate two summary figures: one for combined ROC curves and one for
    combined Precision-Recall curves for all attributes.

    Args:
        y_true (np.ndarray): Ground truth labels (N, K), values {0, 1}.
        logits (np.ndarray): Predicted attribute logits (N, K).
        attribute_names (List[str], optional): List of attribute names for labeling.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing two matplotlib Figure objects:
                                       (roc_figure, pr_figure)
    """
    prob_np = torch.sigmoid(torch.tensor(logits)).detach().cpu().numpy()
    num_attributes = y_true.shape[1]

    roc_fig, roc_ax = plt.subplots(figsize=(9, 8))
    pr_fig, pr_ax = plt.subplots(figsize=(9, 8))

    colors = plt.cm.get_cmap("tab10").colors

    for k in range(num_attributes):
        y_true_k = y_true[:, k]
        prob_k = prob_np[:, k]

        if np.unique(y_true_k).size < 2:
            print(
                f"Skipping attribute '{attribute_names[k] if attribute_names else k}' (only one class present in y_true)"
            )
            continue

        fpr, tpr, _ = roc_curve(y_true_k, prob_k)
        precision, recall, _ = precision_recall_curve(y_true_k, prob_k)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        attr_name = attribute_names[k] if attribute_names else f"Attribute {k}"
        color = colors[k % len(colors)]

        roc_ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{attr_name} (AUC = {roc_auc:.3f})"
        )
        pr_ax.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{attr_name} (AP = {pr_auc:.3f})",
        )

    roc_ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.7, label="Chance")
    roc_ax.set_xlabel("False Positive Rate", fontsize=12)
    roc_ax.set_ylabel("True Positive Rate", fontsize=12)
    roc_ax.set_title(
        "Combined Receiver Operating Characteristic (ROC) Curves", fontsize=14
    )
    roc_ax.legend(loc="lower right", fontsize="small")
    roc_ax.grid(alpha=0.3)
    roc_fig.tight_layout()

    global_prevalence = np.sum(y_true) / y_true.size
    pr_ax.axhline(
        y=global_prevalence,
        color="k",
        linestyle="--",
        lw=2,
        alpha=0.7,
        label=f"No Skill (Prevalence={global_prevalence:.2f})",
    )
    pr_ax.set_xlabel("Recall", fontsize=12)
    pr_ax.set_ylabel("Precision", fontsize=12)
    pr_ax.set_title("Combined Precision-Recall (PR) Curves", fontsize=14)
    pr_ax.legend(loc="upper right", fontsize="small")
    pr_ax.grid(alpha=0.3)
    pr_fig.tight_layout()

    return roc_fig, pr_fig
