import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


def summarize_capsule_poses(poses: np.ndarray):
    """
    Args:
        - poses (ndarray): array of shape (n_samples, n_capsules, pose_dim).
    Returns:
        Tensor: tensor containing a summary statistic for each capsule pose (n_samples, n_capsules).
    """
    summary = np.linalg.norm(poses, axis=-1)
    return summary


def compute_pairwise_mi(summary: np.ndarray):
    """
    Args:
        - summary (ndarray): summary statistics for the poses of capsules.
    Returns:
        Tensor: MI values (n_capsules, n_capsules).
    """
    n_samples, n_capsules = summary.shape

    scaler = StandardScaler()
    summary = scaler.fit_transform(summary)

    mi_matrix = np.zeros((n_capsules, n_capsules))

    for i in range(n_capsules):
        for j in range(n_capsules):
            mi_matrix[i, j] = mutual_info_regression(summary[:, [i]], summary[:, j])[0]

    return mi_matrix


def plot_mi_heatmap(
    mi_matrix, title="Pairwise Mutual Information Matrix", filename="figures/pmi.pdf"
):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        mi_matrix,
        annot=True,
        fmt=".3f",
        cmap="BuPu",
        cbar_kws={"label": "Mutual Information"},
    )

    plt.title(title)
    plt.xlabel("Capsule j")
    plt.ylabel("Capsule i")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath)
    print(f"Saved figure at path {filepath}")
    plt.close()
