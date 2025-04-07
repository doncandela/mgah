#!/bin/bash
# cc-expts-unity/mx2-unity.sh 4/7/25 D.C.
# sbatch script to run granular-memory simulation program mx2.py non-containerized
# on the Unity cluster, as an example for "My cheat sheet for MPI, GPU, Apptainer,
# and HPC".
#
# Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
# Reads default config file mx2.yaml in grandparent directory modified by
# mx2mod.yaml in current directory.

#SBATCH -n 15                        # run 15 MPI ranks (cores here)
#SBATCH -N 1                         # use one node
#SBATCH --mem=100G                   # allocate 100G of memory per node
##SBATCH --exclusive                  # don't share nodes with other jobs
##SBATCH --mem=0                      # allocate all available memory on nodes used
#SBATCH -t 10:00:00                  # time limit 10 hrs (default is 1 hr)
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity

echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load conda/latest             # need this to use conda commands
conda activate dem21                 # environment with OpenMPI, dem21, and dependencies
export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
mpirun --display bindings python ../../mx2.py mx2mod
