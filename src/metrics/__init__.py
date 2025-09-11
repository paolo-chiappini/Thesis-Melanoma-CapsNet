from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from .mig import compute_mig_score
from .pairwise_mutual_information import (
    compute_pairwise_mi,
    plot_mi_heatmap,
    summarize_capsule_poses,
)

__all__ = [
    "summarize_capsule_poses",
    "compute_pairwise_mi",
    "plot_mi_heatmap",
    "compute_mig_score",
]


def build_metrics(metrics_config, num_attributes, device="cuda"):
    metrics_classes = {
        "f1": BinaryF1Score,
        "precision": BinaryPrecision,
        "recall": BinaryRecall,
    }
    metrics = {}

    for m in metrics_config:
        metric_cls = metrics_classes[m["type"]]
        if m.get("per_attribute", False):
            # create metric for each attribute
            for i in range(num_attributes):
                metrics[f"{m['type']}_attr_{i}"] = metric_cls().to(device)
        else:
            # create aggregate metric
            metrics[m["type"]] = metric_cls().to(device)
    return metrics
