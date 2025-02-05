#!/bin/bash
# app.sh 2/5/25 D.C.
# One-task sbatch script using runs an Apptainer container that
# doesn't use MPI or a GPU.
#SBATCH -c 6                  # use 6 CPU cores
#SBATCH -p cpu                # submit to partition cpu

module purge                  # unload all modules
module load apptainer/latest

# run gputest.py in a container without CuPy, sending its output to a file
apptainer exec pack.sif python gputest.py > app-nogpu.out
