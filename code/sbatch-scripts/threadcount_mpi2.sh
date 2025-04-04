#!/bin/bash
# threadcount_mpi2.sh 2/28/25 D.C.
# sbatch script to run threadcount_mpi.py, which uses MPI to time matrix
# multiplications in parallel in several MPI tasks.
# threadcount_mpi2.sh has been modified from threadcount_mpi.sh so NumPy can
# use 2 threads.
#SBATCH -n 4                         # run 4 MPI ranks
#SBATCH -c 2                         # give each rank two cores
#SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate m4p                   # environment with OpenMPI, mpi4py, NumPy and SciPy
mpirun --display bindings --cpus-per-rank 2 python threadcount_mpi.py
