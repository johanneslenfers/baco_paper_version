#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=8           # the number cpus per task
#SBATCH --partition=gpua100      # on which partition to submit the job
#SBATCH --time=2-06:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --gres=gpu:1                # number of gpus assigned
#SBATCH --array=0-19

#SBATCH --job-name=evaluation_mdh_conv_n1       # the name of your job
#SBATCH --mail-type=ALL                         # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=hohlmeye@uni-muenster.de    # your mail address
 
# LOAD MODULES HERE IF REQUIRED
source ~/myModules
# START THE APPLICATION
cd ~/evaluation/
cp -rf baco_paper_version ./mdh_conv_n1/baco_$SLURM_ARRAY_TASK_ID
cd mdh_conv_n1/baco_$SLURM_ARRAY_TASK_ID
python3 mdh_conv_run_n1.py $SLURM_ARRAY_TASK_ID
