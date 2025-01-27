#!/bin/bash
# app-nogpu.sh 1/16/14 D.C.
# Sample one-task sbatch script using a container, but not a GPU
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -p cpu                # submit to partition cpu

module purge                  # unload all modules
module load apptainer/latest

# run gputest.py in a container without CuPy, sending its output to a file
apptainer exec pack.sif python gputest.py > app-nogpu.out
