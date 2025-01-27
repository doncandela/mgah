#!/bin/bash
# app-gpu.sh 1/16/14 D.C.
# Sample one-task sbatch script using a container and a GPU
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -G 1                  # use one GPU
#SBATCH -p gpu                # submit to partition gpu

module purge                  # unload all modules
module load apptainer/latest
module load cuda/12.6         # need CUDA to use a GPU

# run gputest.py in a container with CuPy, sending its output to a file
apptainer exec --nv gputest.sif python gputest.py > app-gpu.out
