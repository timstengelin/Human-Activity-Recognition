import logging
import gin

from train import Trainer
import input_pipeline.datasets as datasets
import models.architectures as architectures
from utils import utils_misc

import wandb

@gin.configurable
def tune(run_paths, key):
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
    parameters_dict = {
        'steps': {
            'min': 500,
            'max': 5000
        },
        'lr_rate': {
            'min': 0.00001,
            'max': 0.01
        },
        'drop_rate': {
            'min': 0.1,
            'max': 0.5
        },
        'model': {
            'values': ["LSTM_model", "GRU_model", "bidi_LSTM_model"]
        },
        'window_size': {
            'values': [125, 250, 375, 500]
        },
        'window_shift': {
            'values': [50, 75, 100, 125]
        },
        'batch_size': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'units': {
            'values': [8, 16, 32, 64]
        }
    }
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

            if config.model == "LSTM_model":
                model = architectures.lstm_architecture(input_shape=feature_shape, n_classes=label_shape[-1],
                                                        dropout_rate=config.drop_rate, units=int(config.units))
            elif config.model == "GRU_model":
                model = architectures.gru_architecture(input_shape=feature_shape, n_classes=label_shape[-1],
                                                       dropout_rate=config.drop_rate, units=int(config.units))
            elif config.model == "RNN_model":
                model = architectures.rnn_architecture(input_shape=feature_shape, n_classes=label_shape[-1],
                                                       dropout_rate=config.drop_rate, units=int(config.units))
            trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val,
                              run_paths=run_paths, total_steps=config.steps, tuning=True, learning_rate=config.lr_rate)
            for _ in trainer.train():
                continue

    # use agent for optimization
    wandb.agent(sweep_id, function=tuning, count=50)

    # clear wandb job
    wandb.finish()