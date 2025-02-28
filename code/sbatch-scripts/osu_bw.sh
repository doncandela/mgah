#!/bin/bash
# osu_bw.sh 2/27/25 D.C.
# sbatch script to run osu_bw.py, which times the speed of MPI messaging
# between two MPI ranks.  As wrt
#SBATCH -n 2                         # run 4 MPI ranks
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate ompi5                 # environment with OpenMPI, NumPy and SciPy
mpirun --display bindings python osu_bw.py
