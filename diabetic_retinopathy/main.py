import gin
import logging

from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from tune_wandb import *
from ensemble_learning import *

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'evaluate',
                    'The different modes are: train, create_ensemble_model, tune, evaluate')

def main(argv):

    # Set the model to be worked with
    #model_names = ['EfficientNetB0']
        # e.g., for mode train, tune or evaluate
    #model_names = ['MobileNetV2', 'EfficientNetB0', 'EfficientNetB3_pretrained', 'DenseNet201_pretrained']
        # e.g., for mode create_ensemble_model
    model_names = ['MobileNetV2_AND_EfficientNetB0_AND_EfficientNetB3_AND_DenseNet201']
        # e.g., for mode evaluate

    # Create empty lists
    models = []

    # Collect data from gin config file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    # Generate folder structures
    run_paths = utils_params.gen_run_folder()

    # Set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # Set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # Call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test = datasets.load()

    # Define models
    for model_name in model_names:
        if model_name == 'LeNet':
            models.append(le_net(input_shape=(256, 256, 3), n_classes=2))
        elif model_name == 'MobileNetV2':
            models.append(mobilenet_v2(input_shape=(256, 256, 3), n_classes=2))
        elif model_name == 'EfficientNetB0':
            models.append(efficientnet_b0(input_shape=(256, 256, 3), n_classes=2))
        elif model_name == 'EfficientNetB3_pretrained':
            models.append(efficientnet_b3_pretrained(input_shape=(256, 256, 3), n_classes=2))
        elif model_name == 'DenseNet201_pretrained':
            models.append(densenet201_pretrained(input_shape=(256, 256, 3), n_classes=2))
        elif model_name == 'MobileNetV2_AND_EfficientNetB0_AND_EfficientNetB3_AND_DenseNet201':
            models.append(mobilenet_v2_AND_efficientnet_b0_AND_efficientnet_b3_AND_densenet201(
                input_shape=(256, 256, 3), n_classes=2))

    # Print model summaries
    for model in models:
        model.summary()

    # Train model, create ensemble model, tune model or evaluate model
    if FLAGS.mode == 'train':
        trainer = Trainer(models[0], ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    elif FLAGS.mode == 'create_ensemble_model':
        create_ensemble_model(models, run_paths)
    elif FLAGS.mode == 'tune':
        tune(run_paths, model_names[0])
    elif FLAGS.mode == 'evaluate':
        evaluate(models[0], ds_test, run_paths)

if __name__ == "__main__":
    app.run(main)