import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_mi_heatmap(
    mi_matrix,
    attribute_names=None,
    capsule_names=None,
    figsize=(12, 6),
    save_path="./plots/mi_heatmap.png",
):
    """
    Plots a heatmap for the mutual information matrix.

    Args:
        mi_matrix (np.ndarray): MI values of shape [num_caps_dims, num_attributes].
        attribute_names (list): Names of attributes (columns).
        capsule_names (list): Names of capsules (rows).
        figsize (tuple): Size of the plot.
        save_path (string): If not None, saves the heatmap image to this path.
    """
    plt.figure(figsize=figsize)

    if attribute_names is None:
        attribute_names = [f"Attr_{i}" for i in range(mi_matrix.shape[1])]
    if capsule_names is None:
        capsule_names = [f"Caps_{i}" for i in range(mi_matrix.shape[0])]

    sns.heatmap(
        mi_matrix,
        xticklabels=attribute_names,
        yticklabels=capsule_names,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "MI"},
    )

    plt.title("Mutual Information: Capsules vs Attributes")
    plt.xlabel("Attributes")
    plt.ylabel("Capsule Dimensions")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
