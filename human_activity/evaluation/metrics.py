import tensorflow as tf
import numpy as np

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name='confusion_matrix', n_classes=12, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.binary_confusion_matrix = self.add_weight(name='confusion_matrix', shape=(self.n_classes, self.n_classes), initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # Convert predictions to a 1D array of predicted class indices
        predictions = tf.reshape(tf.math.argmax(predictions, axis=-1), shape=(-1, 1))
        predictions = tf.cast(predictions, dtype=tf.float32)
        labels = tf.reshape(tf.math.argmax(labels, axis=-1), shape=(-1, 1))


        # Compute the confusion matrix and update the state
        self.binary_confusion_matrix= tf.math.confusion_matrix(
            tf.squeeze(labels),
            tf.squeeze(predictions),
            num_classes=self.n_classes,
            dtype=tf.float32)

    def result(self):
        return self.binary_confusion_matrix

class Categorical_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(Categorical_Accuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.accuracy = self.add_weight(name='categorical_accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # get highest rated class
        labels = tf.argmax(labels, axis=-1).numpy().flatten()
        predictions = tf.argmax(predictions, axis=-1).numpy().flatten()

        # delete the samples without classification
        index = np.argwhere(labels == 0)
        label_clean = np.delete(labels, index, axis=0)
        predictions_clean = np.delete(predictions, index, axis=0)

        # compare true and predicted
        acc = tf.cast(tf.equal(label_clean, predictions_clean), dtype=tf.float32)

        # reduce to one value entry
        self.accuracy = tf.reduce_mean(acc)

    def result(self):
        # return parameters
        return self.accuracy