import gin
import tensorflow as tf

import models.layers as custom_layers

@gin.configurable
def lstm_architecture(input_shape, n_classes, dropout_rate):
    """A basic LSTM architecture

        Parameters:
            inputs (list): input dimensions of the model
            n_classes (int): number of one-hot encoded classes

        Returns:
            (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.basic_lstm_layer(inputs=inputs, units=250, return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=125)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_lstm_layer(inputs=inputs, units=125, return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=64)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_lstm_layer(inputs=out, units=64, return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=32)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes, activation="softmax")

    return tf.keras.Model(inputs=inputs, outputs=out, name='LSTM_model')S