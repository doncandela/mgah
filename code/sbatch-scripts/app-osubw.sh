#!/bin/bash
# app-osubw.sh 3/1/25 D.C.
# Two-task sbatch script uses an Apptainer container ompi5.sif that
# has OpenMPI to rub osu_bw.py, which measures the commnunication
# speed between two MPI ranks.
# Must set SIFS to directory containing ompi5.sif before running this
# script in a directory containing osu_bw.py
#SBATCH -n 2                       # allocate for two MPI ranks
#SBATCH -p cpu                     # submit to partition cpu
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
# mpirun will run container gpu.sif in two ranks; in each rank
# python in container will run osu_bw.py in CWD.
mpirun --display bindings \
    apptainer exec "$SIFS"/gpu.sif python osu_bw.py
