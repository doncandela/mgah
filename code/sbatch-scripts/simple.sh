#!/bin/bash
# simple.sh 2/5/25 D.C.
# One-task sbatch script using none of MPI, a GPU, or Apptainer.
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -p cpu                # submit to partition cpu

module purge                  # unload all modules
module load conda/latest
conda activate npsp           # environment with NumPy and SciPy but not CuPy

python gputest.py > noapp-nogpu.out   # run gputest.py sending its output to a file
