import gin
import tensorflow as tf

from models.layers import *


@gin.configurable
def mobilenet_v2(input_shape, n_classes, alpha=1.0, dropout_rate=0.2):
    '''
    Defines the MobileNetV2 architecture

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes
        alpha (float): Width multiplier to control the number of filters
        dropout_rate (float): Dropout rate for the top layer

    Returns:
        (tf.keras.Model): MobileNetV2 model
    '''
    def round_filters(filters, alpha):
        return max(8, int(filters * alpha + 0.5) // 8 * 8)

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # Initial Conv2D layer
    x = tf.keras.layers.Conv2D(round_filters(32, alpha), kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Bottleneck blocks
    x = bottleneck_block(x, filters=16, expansion_factor=1, stride=1, repeats=1, alpha=alpha)
    x = bottleneck_block(x, filters=24, expansion_factor=6, stride=2, repeats=2, alpha=alpha)
    x = bottleneck_block(x, filters=32, expansion_factor=6, stride=2, repeats=3, alpha=alpha)
    x = bottleneck_block(x, filters=64, expansion_factor=6, stride=2, repeats=4, alpha=alpha)
    x = bottleneck_block(x, filters=96, expansion_factor=6, stride=1, repeats=3, alpha=alpha)
    x = bottleneck_block(x, filters=160, expansion_factor=6, stride=2, repeats=3, alpha=alpha)
    x = bottleneck_block(x, filters=320, expansion_factor=6, stride=1, repeats=1, alpha=alpha)

    # Last Conv2D layer
    x = tf.keras.layers.Conv2D(round_filters(1280, alpha), kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Identity layer for deep visualization
    x = tf.keras.layers.Activation(activation='linear', name='last_conv')(x)

    # Global average pooling, dropout, and dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='mobilenet_v2')



@gin.configurable
def efficientnet_b0(input_shape, n_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
    '''
    Defines the EfficientNetB0 architecture

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes
        width_coefficient (float): Width scaling factor
        depth_coefficient (float): Depth scaling factor
        dropout_rate (float): Dropout rate for the top layer

    Returns:
        (tf.keras.Model): EfficientNetB0 model
    '''
    def round_filters(filters, width_coefficient):
        return max(8, int(filters * width_coefficient + 0.5) // 8 * 8)

    def round_repeats(repeats, depth_coefficient):
        return int(repeats * depth_coefficient + 0.5)

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # Initial Conv2D layer
    x = tf.keras.layers.Conv2D(round_filters(32, width_coefficient), kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # MBConv blocks
    x = mbconv_block(x, filters=16, expansion_factor=1, stride=1, repeats=round_repeats(1, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=24, expansion_factor=6, stride=2, repeats=round_repeats(2, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=40, expansion_factor=6, stride=2, repeats=round_repeats(2, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=80, expansion_factor=6, stride=2, repeats=round_repeats(3, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=112, expansion_factor=6, stride=1, repeats=round_repeats(3, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=192, expansion_factor=6, stride=2, repeats=round_repeats(4, depth_coefficient), width_coefficient=width_coefficient)
    x = mbconv_block(x, filters=320, expansion_factor=6, stride=1, repeats=round_repeats(1, depth_coefficient), width_coefficient=width_coefficient)

    # Last Conv2D layer
    x = tf.keras.layers.Conv2D(round_filters(1280, width_coefficient), kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Identity layer for deep visualization
    x = tf.keras.layers.Activation(activation='linear', name='last_conv')(x)

    # Global average pooling, dropout, and dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_b0')


@gin.configurable
def efficientnet_b3_pretrained(input_shape, n_classes, trainable_rate=0.2, dropout_rate=0.2):
    '''
    Defines a pretrained EfficientNetB3 architecture

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes
        trainable_rate (float): proportion of trainable parameters in the feature extraction module
        dropout_rate (float): Dropout rate for the top layer

    Returns:
        (tf.keras.Model): EfficientNetB3 model
    '''

    # Define input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess input data
    preprocessed_inputs = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # Build EfficientNetB3 model
    base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape,
                                                      pooling=None, classifier_activation=None)

    # Determine layers to freeze
    last_nontrainable_layer = len(base_model.layers) - int(len(base_model.layers) * trainable_rate)

    # Freeze layers up to the fine-tune index
    for lay in base_model.layers[:last_nontrainable_layer]:
        lay.trainable = False

    # Pass preprocessed inputs through base model
    x = base_model(preprocessed_inputs)

    # Apply identity activation
    x = tf.keras.layers.Activation(activation='linear', name='last_conv')(x)

    # Global average pooling and dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_b3_pretrained')


@gin.configurable
def densenet201_pretrained(input_shape, n_classes, trainable_rate=0.2, dropout_rate=0.2):
    '''
    Defines a pretrained DenseNet201 architecture

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes
        trainable_rate (float): proportion of trainable parameters in the feature extraction module
        dropout_rate (float): Dropout rate for the top layer

    Returns:
        (tf.keras.Model): DenseNet201 model
    '''

    # Define input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess input data
    preprocessed_inputs = tf.keras.applications.densenet.preprocess_input(inputs)

    # Build DenseNet201 model
    base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape,
                                                   pooling=None, classifier_activation=None)

    # Determine layers to freeze
    last_nontrainable_layer = len(base_model.layers) - int(len(base_model.layers) * trainable_rate)

    # Freeze layers up to the fine-tune index
    for lay in base_model.layers[:last_nontrainable_layer]:
        lay.trainable = False

    # Pass preprocessed inputs through base model
    x = base_model(preprocessed_inputs)

    # Apply identity activation
    x = tf.keras.layers.Activation(activation='linear', name='last_conv')(x)

    # Global average pooling and dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='densenet201_pretrained')


@gin.configurable
def mobilenet_v2_AND_efficientnet_b0_AND_efficientnet_b3_AND_densenet201(input_shape, n_classes):
    '''
    Defines an ensemble of MobileNetV2, EfficientNetB0, EfficientNetB3 and DenseNet201

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels)
        n_classes (int): Number of output classes

    Returns:
        (tf.keras.Model): ensemble model
    '''

    # Define models
    models = [mobilenet_v2(input_shape=(256, 256, 3), n_classes=2),
              efficientnet_b0(input_shape=(256, 256, 3), n_classes=2),
              efficientnet_b0(input_shape=(256, 256, 3), n_classes=2),
              densenet201_pretrained(input_shape=(256, 256, 3), n_classes=2)]

    # Input layer of ensemble model
    ensemble_input = tf.keras.Input(shape=input_shape)

    # Combine outputs from all models
    model_outputs = [model(ensemble_input) for model in models]

    # Compute the average probabilities across all models
    mean_probs = tf.reduce_mean(model_outputs, axis=0)

    # Determine the final class based on argmax (voting principle)
    final_classes = tf.argmax(mean_probs, axis=1)

    # Convert the final classes to probabilities (100% for the chosen class, 0% for the other class)
    combined_output = tf.one_hot(final_classes, depth=2, dtype=tf.float32)

    return tf.keras.Model(inputs=ensemble_input,
                          outputs=combined_output,
                          name='mobilenet_v2_AND_efficientnet_b0_AND_efficientnet_b3_AND_densenet201')