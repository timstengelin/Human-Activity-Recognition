import logging
import gin
import wandb

from train import Trainer
from input_pipeline import datasets
from models.architectures import *

wandb.login(key="c9cea4bca2336afb9f6eb8e774600fc4ec87b11a")

def tune(run_paths):
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'acc_val',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'total_steps': {
            'min': 500,
            'max': 5000
        },
        'learning_rate': {
            'min': 0.0000001, # 1e-7
            'max': 0.001 # 1e-3
        },
        'trainable_rate': {
            'min': 0.1,
            'max': 0.5
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.5
        },
        'model': {
            'values': ['ResNet50_pretrained'] #['MobileNetV2_pretrained'] #["DenseNet201_pretrained"] # ["LSTM_model", "GRU_model", "RNN_model"]
         },
        'batch_size': {
            'min': 8,
            'max': 64
        }
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep=sweep_config, project="resnet50_pretrained")



    def func():
        config = {
            'total_steps': 3000,
            'learning_rate': 0.0001,
            'trainable_rate': 0.2,
            'dropout_rate': 0.2,
            'batch_size': 16,
        }
        if config:
            run = wandb.init(config=config)

            config = wandb.config

            # Model training code here ...
            ds_train, ds_val, ds_test = datasets.load(load_record=False, batch_size=config.batch_size, augmentation=True)

            if config.model == 'LeNet':
                model = le_net(input_shape=(256, 256, 3),
                               n_classes=2)
            elif config.model == 'MobileNetV2':
                model = mobilenet_v2(input_shape=(256, 256, 3),
                                     n_classes=2,
                                     alpha=config.alpha,
                                     dropout_rate=config.dropout_rate)
            elif config.model == 'EfficientNetB0':
                model = efficientnet_b0(input_shape=(256, 256, 3),
                                        n_classes=2,
                                        width_coefficient=config.width_coefficient,
                                        depth_coefficient=config.depth_coefficient,
                                        dropout_rate=config.dropout_rate)
            elif config.model == 'MobileNetV2_pretrained':
                model = mobilenet_v2_pretrained(input_shape=(256, 256, 3),
                                                n_classes=2,
                                                trainable_rate=config.trainable_rate,
                                                dropout_rate=config.dropout_rate)
            elif config.model == 'DenseNet201_pretrained':
                model = densenet201_pretrained(input_shape=(256, 256, 3),
                                               n_classes=2,
                                               trainable_rate=config.trainable_rate,
                                               dropout_rate=config.dropout_rate)
            elif config.model == 'ResNet50_pretrained':
                model = resnet50_pretrained(input_shape=(256, 256, 3),
                                               n_classes=2,
                                               trainable_rate=config.trainable_rate,
                                               dropout_rate=config.dropout_rate)

            trainer = Trainer(model=model,
                              ds_train=ds_train,
                              ds_val=ds_val,
                              run_paths=run_paths,
                              total_steps=config.total_steps,
                              learning_rate=config.learning_rate,
                              tuning=True)
            for _ in trainer.train():
                continue

    wandb.agent(sweep_id, function=func, count=50)

    wandb.finish()