import gin
import tensorflow as tf
import logging
import wandb
import evaluation.metrics as metrics
import numpy as np

weight=np.array([1,1,1,1,1,1,5,5,5,5,5,5])

def crossentropy(labels, predictions, weights, *args, **kwargs):
    # scale predictions so that the class probas of each sample sum to 1
    predictions /= tf.keras.backend.sum(predictions, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    predictions = tf.keras.backend.clip(predictions, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    # calc
    loss = labels * tf.keras.backend.log(predictions) * weights
    loss = -tf.keras.backend.sum(loss, -1)

    return loss

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, learning_rate, run_paths,
                 total_steps, log_interval, ckpt_interval, tuning=False):
        # Loss objective
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # self.loss_object = crossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.train_accuracy = metrics.Categorical_Accuracy()

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        # self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        self.val_accuracy = metrics.Categorical_Accuracy()

        # attributes
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.tuning = tuning

        # Checkpoint Manager
        if not tuning:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
            self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt, directory=self.run_paths["path_ckpts_train"],
                                                           max_to_keep=10)

        # Summary Writer
        if not tuning:
            self.train_summary_writer = tf.summary.create_file_writer(self.run_paths['path_board_train'])
            self.val_summary_writer = tf.summary.create_file_writer(self.run_paths['path_board_val'])

    @tf.function
    def train_step(self, data, labels):
        """A basic single LSTM layer

            Parameters:
                data (Tensor): input data for train step
                labels (int): true labels for train step

            Returns:
                none
        """
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            # loss = self.loss_object(labels, predictions, sample_weight=sample_weight)
            loss = crossentropy(labels, predictions, weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, data, labels):
        """A basic single LSTM layer

            Parameters:
                data (Tensor): input data for validation step
                labels (int): true labels for validation step

            Returns:
                none
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        # t_loss = self.loss_object(labels, predictions, sample_weight=sample_weight)
        t_loss = crossentropy(labels, predictions, weight)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        """train function

            Parameters:
                none
            Returns:
                none
        """
        # resume the model by continuing training if model is available
        if not self.tuning:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                logging.info("Restored training data from {}".format(self.ckpt_manager.latest_checkpoint))
                self.ckpt.step.assign_add(1)
            else:
                logging.info("Starting training for a new model...")

        for idx, (data, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(data, labels)

            # Write summary to tensorboard
            if not self.tuning:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result() * 100, step=step)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_data, val_labels in self.ds_val:
                    self.val_step(val_data, val_labels)

                template = 'Step {}, Loss: {:.6f}, Accuracy: {:.6f}, Validation Loss: {:.6f}, Validation Accuracy: {:.6f}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))
                
                # Write summary to tensorboard
                if not self.tuning:
                    with self.val_summary_writer.as_default():
                        tf.summary.scalar('loss', self.val_loss.result(), step=step)
                        tf.summary.scalar('accuracy', self.val_accuracy.result() * 100, step=step)

                if self.tuning:
                    wandb.log({"acc_train": self.train_accuracy.result()*100,
                               "loss_train": self.train_loss.result(),
                               "acc_val": self.val_accuracy.result()*100,
                               "loss_val": self.val_loss.result(),})

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                # Save checkpoint
                if not self.tuning:
                    self.ckpt_manager.save()
                    logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')

            if step % self.total_steps == 0:
                # Save final checkpoint
                if not self.tuning:
                    self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps.')
                return self.val_accuracy.result().numpy()
