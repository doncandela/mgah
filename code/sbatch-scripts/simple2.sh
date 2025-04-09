#!/bin/bash
# simple2.sh 4/9/25 D.C.
# One-task sbatch script using none of MPI, a GPU, or Apptainer.
# This version writes the batch script to the ouput file.
#SBATCH -c 6                         # use 6 CPU cores
#SBATCH -p cpu                       # submit to partition cpu
scontrol write batch_script $SLURM_JOB_ID -;echo # print this script to output
echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate npsp                  # environment with NumPy and SciPy but not CuPy
python gputest.py                    # run gputest.py, output will be in slurm-<jobid>.out
