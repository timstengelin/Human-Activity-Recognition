import gin
import tensorflow as tf


@gin.configurable
def inverted_residual_block(inputs, filters, expansion_factor, stride, alpha=1.0):
    """
    Defines an inverted residual block with optional expansion.

    Args:
        inputs (Tensor): Input tensor.
        filters (int): Number of output filters.
        expansion_factor (int): Expansion factor for the input channels.
        stride (int): Stride for the depthwise convolution.
        alpha (float): Width multiplier for scaling the number of filters.

    Returns:
        Tensor: Output tensor after applying the inverted residual block.
    """
    input_channels = inputs.shape[-1]
    expanded_channels = input_channels * expansion_factor
    filters = int(filters * alpha)

    x = inputs

    # Expand
    if expansion_factor != 1:
        x = tf.keras.layers.Conv2D(expanded_channels, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Project
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip connection
    if stride == 1 and input_channels == filters:
        x = tf.keras.layers.Add()([inputs, x])

    return x