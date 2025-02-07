import tensorflow as tf


class BinaryAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='binary_accuracy', **kwargs):
        '''
        Initialize the metric with the given name and optional arguments;
        Add two state variables: `total` (total samples) and `correct`
            (correct predictions)
        '''

        super(BinaryAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.correct = self.add_weight(name="correct", initializer="zeros")

    def update_state(self, labels, predictions, *args, **kwargs):
        '''
        Update the state variables based on new batch of labels and predictions
        '''

        # Determine the predicted classes (argmax across the last axis)
        predicted_classes = tf.argmax(predictions, axis=-1)
        # Ensure labels and predictions are of type int64
        labels = tf.cast(labels, tf.int64)
        predicted_classes = tf.cast(predicted_classes, tf.int64)

        # Count the number of correct predictions
        correct_predictions = tf.reduce_sum(
            tf.cast(predicted_classes == labels, tf.float32))

        # Update the state variables
        self.correct.assign_add(correct_predictions)
        self.total.assign_add(tf.cast(tf.size(labels), tf.float32))

    def result(self):
        '''
        Compute and return the binary accuracy as correct/total;
        Handle division by zero using `divide_no_nan`
        '''

        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        '''
        Reset the state variables to their initial values
        '''

        self.total.assign(0)
        self.correct.assign(0)


class BinaryConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name='binary_confusion_matrix', **kwargs):
        '''
        Initialize the metric with the given name and optional arguments;
        Add state variables to track true positives, false positives,
            true negatives, and false negatives
        '''

        super(BinaryConfusionMatrix, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, labels, predictions, *args, **kwargs):
        '''
        Update the state variables based on new batch of labels and predictions
        '''

        # Determine the predicted classes (argmax across the last axis)
        predicted_classes = tf.argmax(predictions, axis=-1)
        # Ensure labels and predictions are of type int64
        labels = tf.cast(labels, tf.int64)
        predicted_classes = tf.cast(predicted_classes, tf.int64)

        # Update true positives, false positives, true negatives, and false
        # negatives
        self.true_positives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    (predicted_classes == 1) & (
                        labels == 1),
                    tf.float32)))
        self.false_positives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    (predicted_classes == 1) & (
                        labels == 0),
                    tf.float32)))
        self.true_negatives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    (predicted_classes == 0) & (
                        labels == 0),
                    tf.float32)))
        self.false_negatives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    (predicted_classes == 0) & (
                        labels == 1),
                    tf.float32)))

    def result(self):
        '''
        Compute and return the confusion matrix as a 2x2 tensor
        '''

        return tf.convert_to_tensor([
            [self.true_negatives, self.false_positives],  # First row: TN, FP
            [self.false_negatives, self.true_positives]  # Second row: FN, TP
        ])

    def reset_states(self):
        '''
        Reset the state variables to their initial values
        '''

        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)
