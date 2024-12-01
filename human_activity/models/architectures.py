import gin
import tensorflow as tf

import models.layers as custom_layers

@gin.configurable
def lstm_architecture(input_shape, n_classes):
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.basic_lstm_layer(inputs=inputs, units=250, return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=125)
    out = custom_layers.basic_lstm_layer(inputs=out, units=64, return_sequences=False)
    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes, activation="sigmoid")
    return tf.keras.Model(inputs=inputs, outputs=out, name='lstm_base')