#!/bin/bash
# app-osubwmod.sh 3/2/25 D.C.
# Two-task sbatch script uses an Apptainer container ompi5.sif that
# has OpenMPI to run osu_bw.py, which measures the commnunication
# speed between two MPI ranks.
# This is the same as app-osubw.mod except that the two ranks are
# forced to be on separate nodes.
# Activates the Conda environment ompi5 to make OpenMPI available
# outside the container.
# Must set SIFS to directory containing ompi5.sif before running this
# script in a directory containing osu_bw.py
#SBATCH -N 2            # allocate two nodes (change from app-osubs.sh)
#SBATCH -n 2                       # allocate for two MPI ranks
#SBATCH -p cpu                     # submit to partition cpu
#SBATCH -C ib                      # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
module load conda/latest
conda activate ompi5
# mpirun will run container gpu.sif in two ranks; in each rank
# python in container will run osu_bw.py in CWD.
# --map-by node is change from app-osubw.sh, forces both nodes to be used.
mpirun --display bindings --map-by node \
    apptainer exec $SIFS/ompi5.sif python osu_bw.py
