#!/bin/bash
#SBATCH --job-name=exp-frcnn-b             # Job name
#SBATCH --output=/nfs/users/ext_cvgroup-9/jgeob/pups/OUTPUT/slurm-logs/Ex1-frcnnBase_%A_%N.txt # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=80G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --time=24:00:00             # Specify the time needed for your experiment
#SBATCH --reservation=cv703         # for partition allocation
#SBATCH --partition=cv703           # for partition allocation

## OR
## srun -N 1 --gres=gpu:1 --time=01:00:00 --reservation=cv703 --partition=cv703 --pty bash

date +"%D %T"

export PYTHONPATH=/nfs/users/ext_cvgroup-9/jgeob/pups/Detectron2-modif/:$PYTHONPATH
python tasks/baseline.py

date +"%D %T"

