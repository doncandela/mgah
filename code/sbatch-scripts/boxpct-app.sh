#!/bin/bash
# boxpct-app.sh 4/6/25 D.C.
# n-task sbatch script uses an Apptainer container dem21.sif that
# has OpenMPI and the dem21 package to run boxpct.py, which is a
# test program for the dem21 simulation package.
# Must set SIFS to directory containing dem21.sif before running this
# script in a directory containing boxpct.py
#SBATCH -n 4                       # allocate for n MPI ranks
#SBATCH -p cpu                     # submit to partition cpu
#SBATCH -C ib                      # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
module load conda/latest
conda activate ompi
export pproc=mpi                   # tells dem21 to run in MPI-parallel mode
mpirun --display bindings apptainer exec $SIFS/dem21.sif python boxpct.py
