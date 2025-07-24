from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import numpy as np
import torch


def compute_capsule_activations(model, dataloader, device="cuda", prepare_batch=None):
    assert (
        prepare_batch is not None
    ), "Must specify a way to unpack the batch data, prepare_batch cannot be None"

    model.eval()
    all_caps = []
    all_attrs = []
    with torch.no_grad():
        for batch in dataloader:
            batch_dict = prepare_batch(batch)
            images = batch_dict["inputs"].to(device)
            attrs = batch_dict["visual_attributes"].cpu().numpy()
            caps = model.encode(images)

            all_caps.append(caps.cpu().numpy())
            all_attrs.append(attrs)
    return np.concatenate(all_caps), np.concatenate(all_attrs)


def mutual_information_capsules(caps_activations, attributes, discrete_attributes=True):
    """
    Computes the mutual information between the capsule activations and the corresponding attributes

    Args:
        caps_activations (Tuple): [N, num_caps * caps_dim]
        attributes (Tuple): [N, num_attributes]
        discrete_attributes (bool, optional): Defaults to True. If True binary attributes, else categorical.
    """
    N, num_attributes = attributes.shape
    caps_flat = caps_activations.reshape(N, -1)

    mi_results = np.zeros((caps_flat.shape[1], num_attributes))
    for i in range(num_attributes):
        y = attributes[:, i]
        if discrete_attributes:
            mi = mutual_info_classif(caps_flat, y, discrete_features=False)
        else:
            mi = mutual_info_regression(caps_flat, y)
        mi_results[:, i] = mi
    return mi_results
