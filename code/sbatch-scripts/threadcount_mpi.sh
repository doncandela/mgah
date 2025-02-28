#!/bin/bash
# threadcount_mpi.sh 2/26/25 D.C.
# sbatch script to run threadcount_mpi.py, which uses MPI to time matrix
# multiplications in parallel in several MPI tasks (which don't communicate
# with each other, so not a good test of communication speed).
#SBATCH -n 4                         # run 4 MPI ranks
#SBATCH -c 2                         # give each rank two cores
#SBATCH --mem-per-cpu=8G             # give each core 8 GB of memory
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
export OMP_NUM_THREADS=2             # tell Numpy to use all cores
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate ompi5                 # environment with OpenMPI, NumPy and SciPy
# Use mpirun to n copies of threadcount_mpi.py. n could be specified here, but
# should default automatically to #SBATCH -n value. Output will go to
# slurm_<jobid>.out.
mpirun --display bindings --cpus-per-rank 2 python threadcount_mpi.py
