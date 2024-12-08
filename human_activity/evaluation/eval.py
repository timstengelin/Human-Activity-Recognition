import tensorflow as tf
import gin
import evaluation.metrics as metrics
import logging
import numpy as np

@gin.configurable
def evaluate(model, ds_test, run_paths, n_classes):
    """evaluate the performance of the model

    Parameters:
        model (keras.Model): keras model object for evaluation
        ds_test (tf.data.Dataset): test set
        model_name (string): name of the model
        run_paths (dictionary): storage path of model information
        n_classes (int): number of one hot encoded labels
    """

    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    step = int(checkpoint.step.numpy())

    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored model from {}".format(checkpoint_manager.latest_checkpoint))

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])#, [ConfusionMatrix(num_categories=num_categories)]])

    print(model.summary())

    # evaluate the model
    result = model.evaluate(ds_test, return_dict=True, batch_size=64)

    # log the evaluation information
    logging.info(f"Evaluating at step: {step}...")
    # for key, value in result.items():
    #     logging.info('{}:\n{}'.format(key, value))

    print(result)

    # for idx, (test_data, test_labels) in enumerate(ds_test):
    #     predictions = model(test_data, training=False)
    #     predictions = tf.argmax(predictions, axis=2)
    #     predictions = np.concatenate(predictions.numpy()).flatten()
    #     test_labels = np.concatenate(test_labels.numpy()).flatten()
    #
    #     # postprocess the predictions by using the median filter or percentile filter
    #     # (also compare the results of filter and choose the best)
    #     plt.figure(dpi=800)
    #     plt.title('POSTPROCESSING METHODS COMPARISON')
    #     plt.xlabel('FILTER SIZE')
    #     plt.ylabel('ACCURACY(%)')
    #     plt.grid(b=True, axis='y')
    #     for percentile in range(45, 60, 5):
    #         size_list = range(0, 255, 10)
    #         acc_list = []
    #         for size in size_list:
    #             if size != 0:
    #                 test_predictions = percentile_filter(predictions, percentile=percentile, size=size)
    #             else:
    #                 test_predictions = predictions
    #             test_accuracy = metrics.accuracy_score(test_labels, test_predictions) * 100
    #             logging.info('accuracy(percentile filter {} with size {}):\n{}'.format(percentile, size, test_accuracy))
    #             acc_list.append(test_accuracy)
    #         if percentile == 50:
    #             plt.plot(size_list, acc_list, marker="s", markersize=3, label=(str(percentile) + '%' + ' Percentile Filter(Median Filter)'))
    #         else:
    #             plt.plot(size_list, acc_list, marker="s", markersize=3, label=(str(percentile) + '%' + ' Percentile Filter'))
    #     plt.legend(loc='lower left')
    #     plt.savefig(run_paths['path_model_id'] + '/logs/eval/postprocessing_plot.png', dpi=800)
    #     plt.show()
    #
    #     # plot the confusion matrix
    #     cm = result['confusion_matrix']
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(cm + 1, norm=colors.LogNorm(vmin=100, vmax=cm.max()), cmap='Wistia')
    #     cbar = ax.figure.colorbar(im, ax=ax)
    #     cbar.ax.set_ylabel('NUMBER OF SAMPLING POINTS', rotation=-90, va="bottom")
    #     ax.set_xticks(np.arange(num_categories))
    #     ax.set_yticks(np.arange(num_categories))
    #     ax.set_xticklabels(['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'])
    #     ax.set_yticklabels(['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'])
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #     plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #     for i in range(num_categories):
    #         for j in range(num_categories):
    #             text = ax.text(j, i, cm[i, j], fontsize='x-small', ha="center", va="center", color="b")
    #     ax.set_title("SEQUENCE TO SEQUENCE CONFUSION MATRIX")
    #     fig.tight_layout()
    #     plt.show()