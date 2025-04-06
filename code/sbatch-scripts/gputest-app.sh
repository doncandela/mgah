#!/bin/bash
# gputest-app.sh 4/6/25 D.C.
# One-task sbatch script uses an Apptainer container gpu.sif
# that has CuPy to run gputest.py, which will use a GPU.
# Must set SIFS to directory containing gpu.sif before running this
# script in a directory containing gputest.py
#SBATCH -c 6                       # use 6 CPU cores
#SBATCH -G 1                       # use one GPU
#SBATCH -p gpu                     # submit to partition gpu
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
module load cuda/12.6              # need CUDA to use a GPU
# Use python in gpu.sif to run gputest.py in CWD; need --nv flag
# on apptainer exec to use a GPU.
apptainer exec --nv $SIFS/gpu.sif python gputest.py
