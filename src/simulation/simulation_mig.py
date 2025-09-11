# SIMULATION
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from metrics import compute_mig_score


def simulate_capsules(entangled=False):
    NUM_SAMPLES = 5000
    NUM_CAPSULES = 5
    NUM_ATTRIBUTES = 5
    POSE_DIM = 16

    true_labels = np.random.randint(0, 2, size=(NUM_SAMPLES, NUM_ATTRIBUTES))
    latent_poses = np.random.randn(NUM_SAMPLES, NUM_CAPSULES, POSE_DIM) * 0.1

    for k in range(NUM_CAPSULES):
        signal = np.random.randn(POSE_DIM) * 2.0
        is_present = true_labels[:, k].astype(bool)
        latent_poses[is_present, k, :] += signal

        if entangled:
            # Introduce entanglement with *multiple* other attributes
            for j in range(NUM_ATTRIBUTES):
                if j == k:
                    continue  # skip self
                entangled_mask = true_labels[:, j].astype(bool)
                entangled_signal = np.random.uniform(-1.0, 1.0, size=POSE_DIM)
                latent_poses[entangled_mask, k, :] += entangled_signal

    results = compute_mig_score(torch.Tensor(latent_poses), torch.Tensor(true_labels))
    return results


def mig_simulation():
    results_ideal = simulate_capsules(entangled=False)

    results_entangled = simulate_capsules(entangled=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.heatmap(
        results_ideal["mi_matrix"],
        annot=True,
        fmt=".3f",
        cmap="BuGn",
        ax=axes[0],
        xticklabels=[f"VA {i}" for i in range(5)],
        yticklabels=[f"Capsule {i}" for i in range(5)],
    )
    axes[0].set_title(f"Ideal (No Entanglement)\nMIG: {results_ideal['mig_score']:.4f}")
    axes[0].set_xlabel("Ground-Truth Visual Attributes")
    axes[0].set_ylabel("Learned Capsule Poses")

    sns.heatmap(
        results_entangled["mi_matrix"],
        annot=True,
        fmt=".3f",
        cmap="OrRd",
        ax=axes[1],
        xticklabels=[f"VA {i}" for i in range(5)],
        yticklabels=[f"Capsule {i}" for i in range(5)],
    )
    axes[1].set_title(f"With Entanglement\nMIG: {results_entangled['mig_score']:.4f}")
    axes[1].set_xlabel("Ground-Truth Visual Attributes")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("../figures/comparison_mig_entanglement_vs_ideal.pdf")
    plt.show()
