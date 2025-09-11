import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mutual_info_score


def robust_mi_estimate(latent_poses: np.ndarray, target: np.ndarray, n_bins=10):
    """
    Robustly estimates MI between a continuous representation and a discrete target.

    1. Summarize the high-D pose vector into a 1D activation score (L2 norm).
    2. Discretize this 1D score by binning (e.g., quantiles).
    3. Compute MI from the resulting contingency table of two discrete variables.

    Args:
        latent_representation (np.ndarray): Shape (num_samples, pose_dim).
        discrete_target (np.ndarray): Shape (num_samples,).
        n_bins (int): Number of bins to use for discretization.

    Returns:
        float: The estimated mutual information score.
    """
    if latent_poses.ndim != 2:
        raise ValueError("latent_representation must be 2D.")

    activation_summary = np.linalg.norm(latent_poses, axis=1)

    binned_activations = pd.qcut(
        activation_summary, q=n_bins, labels=False, duplicates="drop"
    )

    return mutual_info_score(binned_activations, target)


def compute_mig_score(
    latent_poses: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
):
    """
    Calculates the Mutual Information Gap (MIG) score.

    Args:
        latent_poses (ndarray): Extracted capsules poses, shape (num_samples, num_capsules, pose_dim).
        true_labels (ndarray): Ground Truth VA labels, shape (num_samples, num_attributes).

    Returns:
        dict: a dictionary containing the final MIG score, the MI matrix, and the entropies of the labels.
    """
    num_samples, num_capsules, pose_dim = latent_poses.shape
    _, num_attributes = true_labels.shape

    mi_matrix = np.zeros((num_capsules, num_attributes))

    for k in range(num_capsules):
        pose_k = latent_poses[:, k, :]

        for j in range(num_attributes):
            target_j = true_labels[:, j]
            mi_matrix[k, j] = robust_mi_estimate(pose_k, target_j, n_bins=n_bins)

    entropies = np.zeros(num_attributes)
    for j in range(num_attributes):
        p = np.mean(true_labels[:, j])
        if p == 0 or p == 1:
            entropies[j] = 0
        else:
            entropies[j] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    entropies += 1e-8  # avoid div by 0

    sorted_mi = np.sort(mi_matrix, axis=1)[:, ::-1]  # sort descending
    gaps = sorted_mi[:, 0] - sorted_mi[:, 1]

    highest_mi_indices = np.argmax(mi_matrix, axis=1)
    normalized_gaps = gaps / entropies[highest_mi_indices]
    mig_score = np.mean(normalized_gaps)

    return {"mig_score": mig_score, "mi_matrix": mi_matrix, "entropies": entropies}


def plot_mig_heatmap(results, filename="/figures/mig.pdf"):
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    matrix = results["mi_matrix"]

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="BuPu",
        ax=axes,
        xticklabels=[f"VA {i}" for i in range(matrix.shape[0])],
        yticklabels=[f"Capsule {i}" for i in range(matrix.shape[1])],
    )
    axes.set_title(f"MIG: {results['mig_score']:.4f}")
    axes.set_xlabel("Ground-Truth Visual Attributes")
    axes.set_ylabel("Learned Capsule Poses")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
