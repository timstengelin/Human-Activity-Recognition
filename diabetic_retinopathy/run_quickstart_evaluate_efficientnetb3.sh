#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=dr_g15
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.8
# Run your python code
python main.py "['EfficientNetB3_pretrained']" 'evaluate' 'efficientnetb3' 'False'
# Parameter: main_logic.model_names, main_logic.mode, gen_run_folder.path_model_id, gen_run_folder.new_model