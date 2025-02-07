import ast

from absl import app, flags
from evaluation.eval import evaluate
from utils import utils_params, utils_misc
from tune_wandb import *
from ensemble_learning import *

FLAGS = flags.FLAGS


@gin.configurable
def main_logic(run_paths, model_names, mode):
    # Create empty lists
    models = []

    # Set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # Set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # Call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test = datasets.load()

    # Define models
    for model_name in model_names:
        if model_name == 'MobileNetV2':
            models.append(mobilenet_v2())
        elif model_name == 'EfficientNetB0':
            models.append(efficientnet_b0())
        elif model_name == 'EfficientNetB3_pretrained':
            models.append(efficientnet_b3_pretrained())
        elif model_name == 'DenseNet201_pretrained':
            models.append(densenet201_pretrained())
        elif model_name == 'MobileNetV2_AND_EfficientNetB0_AND_EfficientNetB3_AND_DenseNet201':
            models.append(mobilenet_v2_AND_efficientnet_b0_AND_efficientnet_b3_AND_densenet201())

    # Print model summaries
    for model in models:
        model.summary()

    # Train model, create ensemble model, tune model or evaluate model
    if mode == 'train':
        trainer = Trainer(models[0], ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    elif mode == 'create_ensemble_model':
        create_ensemble_model(models, run_paths)
    elif mode == 'tune':
        tune(run_paths, model_names[0])
    elif mode == 'evaluate':
        evaluate(models[0], ds_test, run_paths)


def main(argv):
    # Collect data from gin config file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    if len(argv) == 1:
        # Run with manual configuration, represented in config.gin

        # Generate folder structures
        run_paths = utils_params.gen_run_folder()

        main_logic(run_paths)
    else:
        # Run Quickstart

        # Generate folder structures
        run_paths = utils_params.gen_run_folder(argv[3], ast.literal_eval(argv[4]))

        main_logic(run_paths, ast.literal_eval(argv[1]), argv[2])


if __name__ == "__main__":
    app.run(main)
