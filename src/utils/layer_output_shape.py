import torch


def get_network_output_shape(input_shape, layers, print_all=False):
    """
    Utility function to calculate the output shape of a network

    Args:
        input_shape (tuple): The input shape of the network
        layers (list): The list of layers in the network

    Returns:
        int: The output shape of the network
    """
    output_shape = input_shape

    with torch.no_grad():
        x = torch.randn(*input_shape)
        for layer in layers:
            x = layer(x)
            output_shape = x.shape
            if print_all:
                print(f"{layer.__class__.__name__} output shape: {output_shape}")

    return output_shape
