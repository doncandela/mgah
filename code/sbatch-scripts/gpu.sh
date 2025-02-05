#!/bin/bash
# gpu.sh 2/5/24 D.C.
# One-task sbatch script using a GPU but not Apptainer.
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -G 1                  # use one GPU
#SBATCH -p gpu                # submit to partition gpu

module purge                  # unload all modules
module load conda/latest
module load cuda/12.6         # need CUDA to use a GPU
conda activate gpu            # environment with NumPy, SciPy, and CuPy

python gputest.py > npapp-gpu.out   # run gputest.py sending its output to a file
