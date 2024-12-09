import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import le_net, vgg16, mobilenet_v2

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    # Output for each dataset:
    # <_TakeDataset element_spec=(TensorSpec(shape=(None, None, None), dtype=tf.uint8, name=None),
    # TensorSpec(shape=(), dtype=tf.int64, name=None))>
    # Images: 256x256x3; Label: 0/1 (0 -> noDR, 1 -> DR)
    ds_train, ds_val, ds_test = datasets.load()
    ds_info = 0 # Helper

    '''
    # HELPER: Printing of shapes
    for idx, (data, labels) in enumerate(ds_train):
        print(data.shape, labels.shape)
        break
    '''

    # define model
    model_name = 'LeNet'
    if model_name == 'LeNet':
        model = le_net(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'VGG16':
        model = vgg16(input_shape=(256, 256, 3), n_classes=2)
    elif model_name == 'MobileNetV2':
        model = mobilenet_v2(input_shape=(256, 256, 3), n_classes=2)
    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        '''
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)
        '''
if __name__ == "__main__":
    app.run(main)