import os
import datetime
import gin

@gin.configurable
def gen_run_folder(path_model_id, new_model):
    run_paths = dict()

    path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))

    date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')

    if path_model_id == '':
        run_id = 'run_' + date_creation
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        if new_model and os.path.isdir(os.path.join(path_model_root, path_model_id)):
            path_model_id = path_model_id + "_" + date_creation
        run_paths['path_model_id'] = os.path.join(path_model_root, path_model_id)

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'run.log')
    #run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval', 'run.log')
    run_paths['path_summary_train'] = os.path.join(run_paths['path_model_id'], 'summary', 'train')
    run_paths['path_summary_val'] = os.path.join(run_paths['path_model_id'], 'summary', 'val')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    #run_paths['path_ckpts_eval'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'eval')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)
