#!/bin/bash
# boxpct.sh 4/4/25 D.C.
# sbatch script to run boxpct.py, using the dem21 package in MPI-parallel mode.
#SBATCH -n 4                         # run 4 MPI ranks
#SBATCH -N 1                         # all ranks on one node
#SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate dem21                 # environment with OpenMPI, dem21, and dependencies
export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
mpirun --display bindings python boxpct.py
