import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from evaluation.visualization import visualization
from input_pipeline import datasets
from utils import utils_params, utils_misc
import models.architectures as architectures
import tensorflow as tf
import tune as tuning


train = False

def main(argv):
    tune = True

    # collect data from gin config file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    # generate folder structures for all model and logging data
    run_paths = utils_params.gen_run_folder()

    # set logger for training
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # step for single class per sequence (sequence2oneLabel)
    # ds_train = ds_train.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))
    # ds_val = ds_val.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))
    # ds_test = ds_test.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))

    # get shape from actual dataset
    feature_shape = None
    label_shape = None
    for data, label in ds_train:
        feature_shape = data.shape[1:]
        label_shape = label.shape[1:]
        break

    model_name = "bidi_LSTM_model"
    if model_name == "LSTM_model":
        model = architectures.lstm_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
    elif model_name == "bidi_LSTM_model":
        model = architectures.bidi_lstm_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
    elif model_name == "GRU_model":
        model = architectures.gru_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
    elif model_name == "RNN_model":
        model = architectures.rnn_architecture(input_shape=feature_shape, n_classes=label_shape[-1])

    if train and not tune:
        # initialize Trainer class based on given model and datasets
        trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val, run_paths=run_paths)
        for _ in trainer.train():
            continue
    elif tune and not train:
        tuning.tune(run_paths)
    else:
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        evaluate(model=model,
                 ds_test=ds_test,
                 run_paths=run_paths,
                 n_classes=label_shape[-1])
        visualization(model=model,
                     run_paths=run_paths,
                     dataset=ds_test)

if __name__ == "__main__":
    app.run(main)