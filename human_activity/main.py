import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
import models.architectures as architectures
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # step for single class per sequence (sequence2label)
    ds_train = ds_train.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))
    ds_val = ds_val.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))
    ds_test = ds_test.map(lambda x, y: (x, tf.reduce_mean(y, axis=1)))

    # model
    model = architectures.lstm_architecture(input_shape=(250,6), n_classes=12)

    if FLAGS.train:
        trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val, learning_rate=0.00001, run_paths=run_paths)
        for _ in trainer.train():
            continue
    # else:
    #     evaluate(model,
    #              checkpoint,
    #              ds_test,
    #              ds_info,
    #              run_paths)

if __name__ == "__main__":
    app.run(main)