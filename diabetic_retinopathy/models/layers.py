import gin
import tensorflow as tf


@gin.configurable
def bottleneck_block(inputs, filters, expansion_factor, stride, repeats, alpha):
    '''
    Defines a stack of bottleneck blocks

    Args:
        inputs (Tensor): Input tensor
        filters (int): Number of output filters
        expansion_factor (int): Expansion factor for the input channels
        stride (int): Stride for the first depthwise convolution
        repeats (int): Number of times the block should be repeated
        alpha (float): Width multiplier to control the number of filters

    Returns:
        Tensor: Output tensor after applying the bottleneck blocks
    '''
    def round_filters(filters, alpha):
        return max(8, int(filters * alpha + 0.5) // 8 * 8)

    filters = round_filters(filters, alpha)
    x = inputs

    for i in range(repeats):
        if i == 0:
            x = bottleneck(x, filters, expansion_factor, stride)
        else:
            x = bottleneck(x, filters, expansion_factor, 1)

    return x


@gin.configurable
def bottleneck(inputs, filters, expansion_factor, stride):
    '''
    Defines a single bottleneck block with expansion

    Args:
        inputs (Tensor): Input tensor
        filters (int): Number of output filters
        expansion_factor (int): Expansion factor for the input channels
        stride (int): Stride for the depthwise convolution

    Returns:
        Tensor: Output tensor after applying the bottleneck block
    '''
    input_channels = inputs.shape[-1]
    expanded_channels = input_channels * expansion_factor

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


@gin.configurable
def mbconv_block(inputs, filters, expansion_factor, stride, repeats, width_coefficient):
    '''
    Defines a stack of MBConv blocks

    Args:
        inputs (Tensor): Input tensor
        filters (int): Number of output filters
        expansion_factor (int): Expansion factor for the input channels
        stride (int): Stride for the first depthwise convolution
        repeats (int): Number of times the block should be repeated
        width_coefficient (float): Width scaling factor

    Returns:
        Tensor: Output tensor after applying the MBConv blocks
    '''
    def round_filters(filters, width_coefficient):
        return max(8, int(filters * width_coefficient + 0.5) // 8 * 8)

    filters = round_filters(filters, width_coefficient)
    x = inputs

    for i in range(repeats):
        if i == 0:
            x = mbconv(x, filters, expansion_factor, stride if i == 0 else 1)
        else:
            x = mbconv(x, filters, expansion_factor, 1)

    return x


@gin.configurable
def mbconv(inputs, filters, expansion_factor, stride):
    '''
    Defines a single MBConv block with expansion

    Args:
        inputs (Tensor): Input tensor
        filters (int): Number of output filters
        expansion_factor (int): Expansion factor for the input channels
        stride (int): Stride for the depthwise convolution

    Returns:
        Tensor: Output tensor after applying the MBConv block
    '''
    input_channels = inputs.shape[-1]
    expanded_channels = input_channels * expansion_factor

    x = inputs

    # Expand
    if expansion_factor != 1:
        x = tf.keras.layers.Conv2D(expanded_channels, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Project
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip connection
    if stride == 1 and input_channels == filters:
        x = tf.keras.layers.Add()([inputs, x])

    return x
