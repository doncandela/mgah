#!/bin/bash
# simple.sh 2/24/25 D.C.
# One-task sbatch script using none of MPI, a GPU, or Apptainer.
#SBATCH -c 6                         # use 6 CPU cores
#SBATCH -p cpu                       # submit to partition cpu
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
echo cores/node=$SLURM_CPUS_ON_NODE  # get number of cores on each node
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate npsp                  # environment with NumPy and SciPy but not CuPy
python gputest.py > simpleoutput     # run gputest.py sending its output to a file
