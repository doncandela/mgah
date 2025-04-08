#!/bin/bash
# simple2.sh 4/8/25 D.C.
# One-task sbatch script using none of MPI, a GPU, or Apptainer.
# This version writes the batch script to the ouput file.
#SBATCH -c 6                         # use 6 CPU cores
#SBATCH -p cpu                       # submit to partition cpu
echo nodelist=$SLURM_JOB_NODELIST    # print list of nodes used
echo; scontrol write batch_script $SLURM_JOB_ID -; echo
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate npsp                  # environment with NumPy and SciPy but not CuPy
python gputest.py                    # run gputest.py, output will be in slurm-<jobid>.out
