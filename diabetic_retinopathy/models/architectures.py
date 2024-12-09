import gin
import tensorflow as tf

@gin.configurable
def le_net(input_shape, n_classes):
    '''
    Defines a LeNet architecture.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of output classes.

    Returns:
        (tf.keras.Model): LeNet model.

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