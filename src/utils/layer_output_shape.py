from torch import nn


def unpack_params(kernel_size, stride, padding):
    if isinstance(kernel_size, tuple):
        kernel_height, kernel_width = kernel_size
    else:
        kernel_height = kernel_width = kernel_size

    if isinstance(stride, tuple):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    return (
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_h,
        pad_w,
    )


def get_maxpool2d_output_shape(input_shape, kernel_size, stride, padding="valid"):
    """
    Utility function to calculate the output shape of a MaxPool2D layer

    Args:
        input_shape (tuple): The input shape of the layer
        kernel_size (int): The size of the kernel
        stride (int): The stride of the layer
        padding (str): The padding of the layer

    Returns:
        int: The output shape of the layer
    """
    # Transform padding from string to int
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = (kernel_size - 1) // 2
        else:
            raise ValueError(f"Invalid padding type: {padding}")
    batch_size, channels, height, width = input_shape
    out_height = (height - kernel_size + 2 * padding) // stride + 1
    out_width = (width - kernel_size + 2 * padding) // stride + 1
    return (batch_size, channels, out_height, out_width)


def get_conv2d_output_shape(
    input_shape, output_channels, kernel_size, stride, padding="same"
):
    # Calculate the output shape of the convolutional layer
    batch_size, channels, height, width = input_shape

    kernel_height, kernel_width, stride_height, stride_width, padding, _ = (
        unpack_params(kernel_size, stride, padding)
    )

    # Transform padding from string to int
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = (kernel_size - 1) // 2
        else:
            raise ValueError(f"Invalid padding type: {padding}")
    elif isinstance(padding, int):
        padding = padding
    else:
        raise ValueError(f"Invalid padding type: {padding}")

    # Calculate the output height and width
    out_height = (height - kernel_height + 2 * padding) // stride_height + 1
    out_width = (width - kernel_width + 2 * padding) // stride_width + 1

    return (batch_size, output_channels, out_height, out_width)


def get_convtranspose2d_output_shape(
    input_shape, output_channels, kernel_size, stride, padding="same"
):
    """
    Compute the output shape of a ConvTranspose2d (a.k.a. deconvolution or upconvolution).

    Parameters:
        input_shape (tuple): (batch_size, in_channels, height, width)
        output_channels (int): number of output channels
        kernel_size (int or tuple): kernel size
        stride (int or tuple): stride
        padding (int or 'same' or 'valid'): padding
        output_padding (int or tuple): additional size added to one side of the output

    Returns:
        tuple: (batch_size, output_channels, out_height, out_width)
    """
    batch_size, in_channels, height, width = input_shape

    kernel_height, kernel_width, stride_height, stride_width, padding, _ = (
        unpack_params(kernel_size, stride, padding)
    )

    # Handle padding
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = (kernel_height - 1) // 2
        else:
            raise ValueError(f"Invalid padding type: {padding}")
    elif isinstance(padding, int):
        padding = padding
    else:
        raise ValueError(f"Invalid padding value: {padding}")

    out_height = (height - 1) * stride_height - 2 * padding + kernel_height
    out_width = (width - 1) * stride_width - 2 * padding + kernel_width

    return (batch_size, output_channels, out_height, out_width)


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
    for layer in layers:
        if hasattr(layer, "get_output_shape") and callable(
            getattr(layer, "get_output_shape")
        ):
            output_shape = layer.get_output_shape(output_shape)
        elif isinstance(layer, nn.Conv2d):
            output_shape = get_conv2d_output_shape(
                output_shape,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
            )
        elif isinstance(layer, nn.MaxPool2d):
            output_shape = get_maxpool2d_output_shape(
                output_shape, layer.kernel_size, layer.stride, layer.padding
            )
        elif isinstance(layer, nn.ConvTranspose2d):
            output_shape = get_convtranspose2d_output_shape(
                output_shape,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
            )

        if print_all:
            print(f"{layer.__class__.__name__} output shape: {output_shape}")

    return output_shape
