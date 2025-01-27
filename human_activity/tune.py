import logging
import gin

from train import Trainer
import input_pipeline.datasets as datasets
import models.architectures as architectures
from utils import utils_misc
from train import create_model

import wandb

@gin.configurable
def tune(run_paths, key, parameters_dict):
    utils_misc.set_loggers(run_paths['path_logs_tune'], logging.INFO)
    # wandb login with given key
    wandb.login(key=key)

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'acc_val',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    # create wandb conf and obj
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep=sweep_config, project="Activity Recognition HAPT")


    #actual tuning function
    def tuning():
        config = {
            'steps': 50,
            'lr_rate': 0.01,
            'drop_rate': 0.25,
            'window_size': 250,
            'window_shift': 125,
            'batch_size': 64,
            'units': 64
        }
        if config:
            run = wandb.init(config=config)

            config = wandb.config

            # Model training code here ...
            ds_train, ds_val, ds_test, ds_info = datasets.load(window_size=config.window_size,
                                                               window_shift=config.window_shift,
                                                               batch_size=config.batch_size,
                                                               tfrecord_files_exist=False)

            # get shape from actual dataset
            feature_shape = None
            label_shape = None
            for data, label in ds_train:
                feature_shape = data.shape[1:]
                label_shape = label.shape[1:]
                break

            model = create_model(model_name=config.model, feature_shape=feature_shape, label_shape=label_shape)

            trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val,
                              run_paths=run_paths, total_steps=config.steps, tuning=True, learning_rate=config.lr_rate)
            for _ in trainer.train():
                continue

    # use agent for optimization
    wandb.agent(sweep_id, function=tuning, count=50)

    # clear wandb job
    wandb.finish()