import os
import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from evaluation.metrics import BinaryAccuracy, BinaryConfusionMatrix
from deep_visualization import *

def evaluate(model, ds_test, run_paths):

    logging.info(f'Starting evaluation via metrics of following model from {run_paths["path_ckpts_train"]}.')

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=run_paths["path_ckpts_train"],
                                              max_to_keep=1)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        # .expect_partial() to suppress warnings about unused keys at exit time
    step = int(ckpt.step.numpy())

    if ckpt_manager.latest_checkpoint:
        logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # Evaluate model
    binary_accuracy = BinaryAccuracy()
    binary_confusion_matrix = BinaryConfusionMatrix()

    for data, label in ds_test:
        y_pred = model(data)
        binary_accuracy.update_state(label, y_pred)
        binary_confusion_matrix.update_state(label, y_pred)

    # Plot accuracy
    logging.info(f'Calculated binary accuracy for test dataset: {binary_accuracy.result().numpy()}')

    # Plot confusion matrix
    classes = ["NRDR", "RDR"]
    array = binary_confusion_matrix.result().numpy()
    array = np.around(array.astype('float') / (array.sum(axis=1))[:, np.newaxis], decimals=2)
        # normalize confusion matrix
    df_cm = pd.DataFrame(array, index=[i for i in classes],
                         columns=[i for i in classes])
    sn.heatmap(df_cm, annot=True)
    plt.title("Binary confusion matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(run_paths['path_model_id'], 'confusion_matrix.png'))
    plt.show()

    # Generate deep visualizations (for the outputs of the initial 3 images)
    for idx, (test_image, test_label) in enumerate(ds_test):
        visualize(model, test_image, test_label, idx, run_paths)
        if idx >= 1:
            break
