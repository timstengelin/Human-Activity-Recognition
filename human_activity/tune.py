import logging
import gin

# import ray
# from ray import tune

from train import Trainer
from utils import utils_params, utils_misc
import input_pipeline.datasets as datasets
import models.architectures as architectures

import wandb
def tune(run_paths):
    # parameters
    lr_rate = 1e-4
    steps = 1e3
    model_name = "GRU_model"

    wandb.login(key="8478ddb0f2c0978283abcb1e18db08bebd904d3f")

    # 1. Start a W&B run
    run = wandb.init(
        # Set the project where this run will be logged
        project="iss-dl15",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr_rate,
            "steps": steps,
            "model_name": model_name
        },
    )

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
                      run_paths=run_paths, total_steps=steps, tuning=True)
    for _ in trainer.train():
        continue

    wandb.finish()