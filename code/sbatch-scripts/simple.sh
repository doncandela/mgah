#!/bin/bash
# simple.sh 2/27/25 D.C.
# One-task sbatch script runs gputest.py using none of MPI, a GPU, or Apptainer.
#SBATCH -c 6                         # use 6 CPU cores
#SBATCH -p cpu                       # submit to partition cpu
echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate npsp                  # environment with NumPy and SciPy but not CuPy
python gputest.py                    # run gputest.py, output will be in slurm_<jobid>.out
