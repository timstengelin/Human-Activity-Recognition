import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out

def basic_lstm_layer(inputs, units, return_sequences=True, batch_norm=True, activation="tanh"):
    """A basic single LSTM layer

        Parameters:
            inputs (Tensor): input of the LSTM layer
            units (int): number of filters used for the LSTM layer
            return_sequences (bool): whether output is sequence or not (Default is True)
            batch_norm (bool): whether the batch normalization is used or not (Default is True)
            activation (string): which activation function to use (Default is "tanh")

        Returns:
            (Tensor): output of the single LSTM layer
        """
    out = tf.keras.layers.LSTM(units=units, return_sequences=return_sequences)(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out

@gin.configurable
def basic_dense_layer(inputs, units, batch_norm=True, activation="tanh"):
    """A basic single dense layer

    Parameters:
        inputs (Tensor): input of the dense layer
        units (int): number of filters used for the dense layer
        batch_norm (bool): whether the batch normalization is used or not (Default is True)
        activation (bool): activation function to use (Default is "tanh")

    Returns:
        (Tensor): output of the single dense layer
    """

    out = tf.keras.layers.Dense(units, activation='linear')(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out