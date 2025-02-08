import gin
import sys
from train import training
from evaluation.eval import evaluation
from utils import utils_params
import tune as tuning

def main(argv):
    # collect data from gin configuration file
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    # generate folder structures for all model and logging data
    run_paths = utils_params.gen_run_folder()

    # set config path
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if "--train" in argv:
        training(run_paths)
    elif "--evaluate" in argv:
        evaluation(run_paths)
    elif "--tune" in argv:
        tuning.tune(run_paths)
    else:
        print("Invalid arguments")

if __name__ == "__main__":
    app.run(main)