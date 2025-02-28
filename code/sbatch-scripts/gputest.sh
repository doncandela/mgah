#!/bin/bash
# gputest.sh 2/28/25 D.C.
# One-task sbatch script using a GPU but not Apptainer, runs gputest.py
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -G 1                  # use one GPU
#SBATCH -p gpu                # submit to partition gpu
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                  # unload all modules
module load conda/latest
module load cuda/12.6         # need CUDA to use a GPU
conda activate gpu            # environment with NumPy, SciPy, and CuPy
python gputest.py
