import gin
import logging

from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from tune_wandb import *

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'tune', 'Specifies wheater to train, tune or evaluate a model')

def main(argv):
    model_name = 'MobileNetV2_pretrained'

    # collect data from gin config file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test = datasets.load()

    # define model
    if model_name == 'LeNet':
        model = le_net(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'MobileNetV2':
        model = mobilenet_v2(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'EfficientNetB0':
        model = efficientnet_b0(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'MobileNetV2_pretrained':
        model = mobilenet_v2_pretrained(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'DenseNet201_pretrained':
        model = densenet201_pretrained(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'ResNet50_pretrained':
        model = resnet50_pretrained(input_shape=(256, 256, 3), n_classes=2)
    model.summary()

    if FLAGS.mode == 'train':
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    elif FLAGS.mode == 'tune':
        tune(run_paths, model_name)
    else:
        evaluate(model,
                 ds_test,
                 run_paths)

if __name__ == "__main__":
    app.run(main)