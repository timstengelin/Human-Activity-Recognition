import logging
import gin

# import ray
# from ray import tune

from train import Trainer
from utils import utils_params, utils_misc
import input_pipeline.datasets as datasets
import models.architectures as architectures

import wandb

wandb.login(key="8478ddb0f2c0978283abcb1e18db08bebd904d3f")

def tune(run_paths):
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'steps': {
            'values': [50, 100, 150]
        },
        'lr_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep=sweep_config, project="iss-dl15")



    def func():
        config = {
        'steps': 50,
        'lr_rate': 0.01
        }
        if config:
            run = wandb.init(config=config, magic=True)

            config = wandb.config

            model_name = "GRU_model"
            # Model training code here ...
            ds_train, ds_val, ds_test, ds_info = datasets.load()

            # get shape from actual dataset
            feature_shape = None
            label_shape = None
            for data, label in ds_train:
                feature_shape = data.shape[1:]
                label_shape = label.shape[1:]
                break

            if model_name == "LSTM_model":
                model = architectures.lstm_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
            elif model_name == "GRU_model":
                model = architectures.gru_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
            elif model_name == "RNN_model":
                model = architectures.rnn_architecture(input_shape=feature_shape, n_classes=label_shape[-1])
            trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val,
                              run_paths=run_paths, total_steps=config.steps, tuning=True, learning_rate=config.lr_rate)
            for _ in trainer.train():
                continue

    wandb.agent(sweep_id, function=func, count=10)

    wandb.finish()