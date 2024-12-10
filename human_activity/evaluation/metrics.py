import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...

    def update_state(self, *args, **kwargs):
        return None

    def result(self):
        return None

class Categorical_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(Categorical_Accuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.accuracy = self.add_weight(name='categorical_accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # get highest rated class
        labels = tf.argmax(labels, axis=-1)
        predictions = tf.argmax(predictions, axis=-1)

        # compare true and predicted
        acc = tf.cast(tf.equal(labels, predictions), dtype=tf.float32)

        # reduce to one value over all timesteps and entries
        self.accuracy = tf.reduce_mean(acc)

    def result(self):
        # return parameters
        return self.accuracy