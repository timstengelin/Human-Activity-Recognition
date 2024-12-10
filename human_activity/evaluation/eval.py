import tensorflow as tf
import gin
import evaluation.metrics as metrics
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

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

    accuracy = metrics.Categorical_Accuracy()
    conf_matrix = metrics.ConfusionMatrix(n_classes=n_classes)

    for data, label in ds_test:
        y_pred = model(data)
        accuracy.update_state(label, y_pred)
        conf_matrix.update_state(label, y_pred)
        break
    '''
        for batch in ds_test.take(1):  # Nimm einen Batch aus dem Dataset
        print(batch[0].shape)  # Überprüfe die Form der Eingabedaten
        print(batch[1].shape)
    '''
    print(accuracy.result().numpy())
    # TODO: Not working
    # evaluate the model
    # result = model.evaluate(ds_test, return_dict=False, steps=20)

    # log the evaluation information
    # logging.info(f"Evaluating at step: {step}...")
    # for key, value in result.items():
    #     logging.info('{}:\n{}'.format(key, value))

    # print(result)


    classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING", "STAND_TO_SIT",
               "SIT_TO_STAND", "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    # plot the confusion matrix
    array = conf_matrix.result().numpy()
    # normalize confusion matrix
    array = np.around(array.astype('float') / (array.sum(axis=1))[:, np.newaxis], decimals=2)
    plt.title("Confusion matrix")
    df_cm = pd.DataFrame(array, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    # plt.show()
    plt.savefig('foo.png')