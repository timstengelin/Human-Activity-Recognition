import gin
import tensorflow as tf

from models.layers import vgg_block, inverted_residual_block

@gin.configurable
def le_net(input_shape, n_classes):
    '''
    Defines the LeNet architecture

    Args:
        input_shape (tuple): Shape of the input tensor
        n_classes (int): Number of output classes

    Returns:
        (tf.keras.Model): LeNet model
    '''

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.0)(inputs)

    # LeNet layers
    x = tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', padding='valid')(rescale)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(16, (5, 5), activation='tanh', padding='valid')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='tanh')(x)
    x = tf.keras.layers.Dense(84, activation='tanh')(x)

    outputs = tf.keras.layers.Dense(units=n_classes, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')


@gin.configurable
def vgg16(input_shape, n_classes):
    '''
    Defines the VGG16 architecture

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons

    Returns:
        (tf.keras.Model): VGG16 model
    '''

    inputs = tf.keras.Input(shape=input_shape)

    # Normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.0)(inputs)

    # VGG blocks
    x = vgg_block(rescale, filters=64, kernel_size=(3, 3), n_conv_layers=2)     # Block 1
    x = vgg_block(x, filters=128, kernel_size=(3, 3), n_conv_layers=2)          # Block 2
    x = vgg_block(x, filters=256, kernel_size=(3, 3), n_conv_layers=3)          # Block 3
    x = vgg_block(x, filters=512, kernel_size=(3, 3), n_conv_layers=3)          # Block 4
    x = vgg_block(x, filters=512, kernel_size=(3, 3), n_conv_layers=3)          # Block 5

    # Classification head
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg16')


@gin.configurable
def mobilenet_v2(input_shape, n_classes, alpha=1.0):
    '''
    Defines the MobileNetV2 architecture

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes
        alpha (float): Width multiplier for scaling the number of filters in each layer

    Returns:
        (tf.keras.Model): MobileNetV2 model
    '''

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # First layer: Conv2D with stride 2 and 32 filters
    x = tf.keras.layers.Conv2D(int(32 * alpha), kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Inverted residual blocks
    x = inverted_residual_block(x, filters=16, expansion_factor=1, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=24, expansion_factor=6, stride=2, alpha=alpha)
    x = inverted_residual_block(x, filters=24, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=32, expansion_factor=6, stride=2, alpha=alpha)
    x = inverted_residual_block(x, filters=32, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=32, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=64, expansion_factor=6, stride=2, alpha=alpha)
    x = inverted_residual_block(x, filters=64, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=64, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=64, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=96, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=96, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=96, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=160, expansion_factor=6, stride=2, alpha=alpha)
    x = inverted_residual_block(x, filters=160, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=160, expansion_factor=6, stride=1, alpha=alpha)
    x = inverted_residual_block(x, filters=320, expansion_factor=6, stride=1, alpha=alpha)

    # Last convolutional layer
    x = tf.keras.layers.Conv2D(int(1280 * alpha), kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Global average pooling and dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='mobilenet_v2')