import tensorflow as tf

class BinaryAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='binary_accuracy', **kwargs):
        super(BinaryAccuracy, self).__init__(name=name, **kwargs)
        self.binary_accuracy = self.add_weight(name='binary_accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # Convert labels to integer type for consistency
        labels = tf.cast(labels, dtype=tf.int32)

        # Convert predictions to a 1D array of predicted class indices
        predictions = tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1))
        predictions = tf.cast(predictions, dtype=tf.int32)

        # Compute binary accuracy by checking equality between labels and predictions
        self.binary_accuracy = tf.math.reduce_mean(
            tf.cast(tf.math.equal(labels, predictions),dtype=tf.float32))

    def result(self):
        return self.binary_accuracy


class BinaryConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name='binary_confusion_matrix', **kwargs):
        super(BinaryConfusionMatrix, self).__init__(name=name, **kwargs)
        self.binary_confusion_matrix = self.add_weight(name='binary_confusion_matrix', shape=(2, 2), initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # Convert labels to integer type for consistency
        labels = tf.cast(labels, dtype=tf.int32)

        # Convert predictions to a 1D array of predicted class indices
        predictions = tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1))
        predictions = tf.cast(predictions, dtype=tf.int32)

        # Compute the confusion matrix and update the state
        self.binary_confusion_matrix= tf.math.confusion_matrix(
            tf.squeeze(labels),
            tf.squeeze(predictions),
            num_classes=2,
            dtype=tf.int32)

    def result(self):
        return self.binary_confusion_matrix
