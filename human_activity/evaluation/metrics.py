import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...

    def update_state(self, *args, **kwargs):
        return None

    def result(self):
        return None

class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        # self.predictions = tf.reshape(tf.argmax(predictions, axis=2), [-1])
        # self.labels = tf.reshape(labels, [-1])

        self.accuracy = tf.math.reduce_mean(
            tf.cast(tf.math.equal(self.labels, self.predictions), dtype=tf.float32))

    def result(self):
        # return parameters
        return self.accuracy