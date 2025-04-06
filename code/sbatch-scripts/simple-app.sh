#!/bin/bash
# simple-app.sh 4/6/25 D.C.
# One-task sbatch script uses an Apptainer container pack.sif
# that doesn't have CuPy or OpenMPI to run gputest.py (which detects
# CuPy is not present and so doesn't use a GPU).
# Must set SIFS to directory containing pack.sif before running this
# script in a directory containing gputest.py.
#SBATCH -c 6                       # use 6 CPU cores
#SBATCH -p cpu                     # submit to partition cpu
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
# Use python in pack.sif to run gputeset.py in CWD.
apptainer exec $SIFS/pack.sif python gputest.py
