#!/bin/bash
# noapp-nogpu.sh 1/16/14 D.C.
# Sample one-task sbatch script using neither Apptainer nor GPU
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -p cpu                # submit to partition cpu

module purge                  # unload all modules
module load conda/latest
conda activate npsp           # environment with NumPy and SciPy but not CuPy

python gputest.py > noapp-nogpu.out   # run gputest.py sending its output to a file
