import numpy as np


def dominant_capsules_mi(mi_matrix, attribute_names=None, top_k=3):
    """
    Returns the top-k capsule dimensions most associated with each attribute.

    Args:
        mi_matrix (np.ndarray): MI values [num_caps_dims, num_attributes].
        attribute_names (list): List of attribute names.
        top_k (int): Number of top capsules to return per attribute.

    Returns:
        dict: {attribute: [(caps_dim_idx, MI_value), ...]}
    """
    if attribute_names is None:
        attribute_names = [f"Attr_{i}" for i in range(mi_matrix.shape[1])]

    top_caps = {}
    for attr_idx, attr in enumerate(attribute_names):
        sorted_indices = np.argsort(mi_matrix[:, attr_idx])[::-1]
        top_caps[attr] = [
            (int(idx), float(mi_matrix[idx, attr_idx]))
            for idx in sorted_indices[:top_k]
        ]
    return top_caps
