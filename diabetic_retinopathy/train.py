import gin
import tensorflow as tf
import logging

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, learning_rate):
        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Attributes
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Summary writer
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_train'])
        self.val_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_val'])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=self.run_paths["path_ckpts_train"],
                                                       max_to_keep=10)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):

        # If training is interrupted unexpectedly, resume the model from this point and continue training,
        # otherwise, if it's the initial step of training, start from the beginning
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logging.info("Restored from checkpoint {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.step.assign_add(1)
        else:
            logging.info("Initializing from scratch.")


        for idx, (images, labels) in enumerate(self.ds_train):
            step = int(self.ckpt.step.numpy())

            self.train_step(images, labels)

            # Write train summary to tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=step)
                tf.summary.scalar('accuracy', self.train_accuracy.result() * 100, step=step)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))
                
                # Write validation summary to tensorboard
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.val_accuracy.result() * 100, step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if int(self.ckpt.step) % self.ckpt_interval == 0:
                # Save checkpoint
                ckpt_path = self.ckpt_manager.save()
                logging.info(f"Checkpoint saved at step {int(self.ckpt.step)}: {ckpt_path}")

            if step % self.total_steps == 0:
                # Save final checkpoint
                ckpt_path = self.ckpt_manager.save()
                logging.info(f"Final checkpoint saved at step {int(self.ckpt.step)}: {ckpt_path}")
                return self.val_accuracy.result().numpy()

            self.ckpt.step.assign_add(1)
