import tensorflow as tf

class BinaryAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='binary_accuracy', **kwargs):
        super(BinaryAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.correct = self.add_weight(name="correct", initializer="zeros")

    def update_state(self, labels, predictions, *args, **kwargs):
        predicted_classes = tf.argmax(predictions, axis=-1)
        labels = tf.cast(labels, tf.int64)
        predicted_classes = tf.cast(predicted_classes, tf.int64)

        correct_predictions = tf.reduce_sum(tf.cast(predicted_classes == labels, tf.float32))

        self.correct.assign_add(correct_predictions)
        self.total.assign_add(tf.cast(tf.size(labels), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.total.assign(0)
        self.correct.assign(0)


class BinaryConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name='binary_confusion_matrix', **kwargs):
        super(BinaryConfusionMatrix, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, labels, predictions, *args, **kwargs):
        predicted_classes = tf.argmax(predictions, axis=-1)
        labels = tf.cast(labels, tf.int64)
        predicted_classes = tf.cast(predicted_classes, tf.int64)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast((predicted_classes == 1) & (labels == 1), tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast((predicted_classes == 1) & (labels == 0), tf.float32)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast((predicted_classes == 0) & (labels == 0), tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast((predicted_classes == 0) & (labels == 1), tf.float32)))

    def result(self):
        return tf.convert_to_tensor([
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives]
        ])

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)
