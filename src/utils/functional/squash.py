import torch


def squash(s, dim=-1):
    """
    Non-linear squashing of input vectors s_j.
    v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||   (1)

    Params:
    - s:   vector to squash (before activation)
    - dim: dimension over which to take the norm

    Returns:
    - The squashed input vector
    """
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)
