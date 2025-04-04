#!/bin/bash
# osu_bw.sh 4/3/25 D.C.
# sbatch script to run osu_bw.py, which times the speed of MPI messaging
# between two MPI ranks.
#SBATCH -n 2                         # run 2 MPI ranks
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate m4p                   # environment with OpenMPI, mpi4py, NumPy and SciPy
mpirun --display bindings python osu_bw.py
