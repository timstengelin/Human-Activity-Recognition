import logging
import gin
import wandb

from train import Trainer
from input_pipeline import datasets
from models.architectures import *

wandb.login(key="c9cea4bca2336afb9f6eb8e774600fc4ec87b11a")


# Function to get sweep config, metric, and parameter dictionary based on model selection
def get_sweep_config(model_name):
    if model_name == 'MobileNetV2':
        sweep_config = {
            'method': 'grid', # 'grid' for grid search, 'random' for random search
            'metric': {'name': 'acc_val', 'goal': 'maximize'},
            'parameters': {
                'alpha': {'values': [0.75, 1, 1.25]},
                'dropout_rate': {'values': [0.2, 0.4]},
                'learning_rate': {'values': [1e-3, 5e-4, 1e-4]},
                'batch_size': {'values': [8, 16, 32]},
                'total_steps': {'values': [15000]},
                'augmentation': {'values': [False]}
            }
        }
    elif model_name == 'EfficientNetB0':
        sweep_config = {
            'method': 'grid', # 'grid' for grid search, 'random' for random search
            'metric': {'name': 'acc_val', 'goal': 'maximize'},
            'parameters': {
                'width_and_depth_coefficient': {'values': [0.75, 1, 1.25]},
                'dropout_rate': {'values': [0.2, 0.4]},
                'learning_rate': {'values': [1e-3, 5e-4, 1e-4]},
                'batch_size': {'values': [8, 16, 32]},
                'total_steps': {'values': [30000]},
                'augmentation': {'values': [True]}
            }
        }
    elif model_name == 'EfficientNetB3_pretrained':
        sweep_config = {
            'method': 'grid', # 'grid' for grid search, 'random' for random search
            'metric': {'name': 'acc_val', 'goal': 'maximize'},
            'parameters': {
                'trainable_rate': {'values': [1]},
                'dropout_rate': {'values': [0.2]},
                'learning_rate': {'values': [1e-4, 1e-5]},
                'batch_size': {'values': [32]},
                'total_steps': {'values': [7500]},
                'augmentation': {'values': [True]}
            }
        }
    elif model_name == 'DenseNet201_pretrained':
        sweep_config = {
            'method': 'grid', # 'grid' for grid search, 'random' for random search
            'metric': {'name': 'acc_val', 'goal': 'maximize'},
            'parameters': {
                'trainable_rate': {'values': [1]},
                'dropout_rate': {'values': [0.2]},
                'learning_rate': {'values': [1e-4, 1e-5]},
                'batch_size': {'values': [32]},
                'total_steps': {'values': [7500]},
                'augmentation': {'values': [True]}
            }
        }
    else:
        raise ValueError("Unsupported model name")

    return sweep_config


def tune(run_paths, model_name):
    # Get model-specific sweep configuration
    sweep_config = get_sweep_config(model_name)

    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project='{}'.format(model_name))

    def func():
        # Default configuration for tuning
        config = {'total_steps': 1000}

        if config:
            run = wandb.init(config=config)
            config = wandb.config

            # Load dataset
            ds_train, ds_val, ds_test = datasets.load(load_record=False,  # TFRecord files are created
                                                      batch_size=config.batch_size,
                                                      augmentation=config.augmentation)

            # Model selection based on the chosen model name
            if model_name == 'MobileNetV2':
                model = mobilenet_v2(alpha=config.alpha,
                                     dropout_rate=config.dropout_rate)
            elif model_name == 'EfficientNetB0':
                model = efficientnet_b0(width_coefficient=config.width_and_depth_coefficient,
                                        depth_coefficient=config.width_and_depth_coefficient,
                                        dropout_rate=config.dropout_rate)
            elif model_name == 'EfficientNetB3_pretrained':
                model = densenet201_pretrained(trainable_rate=config.trainable_rate,
                                               dropout_rate=config.dropout_rate)
            elif model_name == 'DenseNet201_pretrained':
                model = densenet201_pretrained(trainable_rate=config.trainable_rate,
                                               dropout_rate=config.dropout_rate)

            # Initialize Trainer
            trainer = Trainer(model=model,
                              ds_train=ds_train,
                              ds_val=ds_val,
                              run_paths=run_paths,
                              total_steps=config.total_steps,
                              learning_rate=config.learning_rate,
                              tuning=True)

            # Start training
            for _ in trainer.train():
                continue

    # Start sweep agent
    wandb.agent(sweep_id, function=func, count=200)
    wandb.finish()