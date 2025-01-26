import gin
import logging
from absl import app, flags

from train import training
from evaluation.eval import evaluation
from input_pipeline import datasets
from utils import utils_params, utils_misc
import models.architectures as architectures
import tensorflow as tf
import tune as tuning

@gin.configurable
def train_tune(tune, evaluate, run_paths):
    if tune:
        tuning.tune(run_paths)
    else:
        if evaluate:
            evaluation(run_paths)
        else:
            training(run_paths)

def main(argv):
    # collect data from gin config file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    # generate folder structures for all model and logging data
    run_paths = utils_params.gen_run_folder()

    # set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    train_tune(run_paths=run_paths)

if __name__ == "__main__":
    app.run(main)