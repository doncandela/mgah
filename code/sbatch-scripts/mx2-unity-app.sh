#!/bin/bash
# cc-expts-unity-app/mx2-unity-app.sh 4/6/25 D.C.
# sbatch script to run granular-memory simulation program mx2.py containerized on the
# Unity cluster, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
#
# Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
# Reads default config file mx2.yaml in grandparent directory modified by
# mx2mod.yaml in current directory. Must set SIFS to directory containing
# dem21.sif before running this script.

#SBATCH -n 16                        # run 16 MPI ranks (cores here)
#SBATCH -N 1                         # use one node
#SBATCH --mem=100G                   # allocate 100G of memory per node
##SBATCH --exclusive                  # don't share nodes with other jobs
##SBATCH --mem=0                      # allocate all available memory on nodes used
#SBATCH -t 120                       # time limit 2 hrs (default is 1 hr)
#SBATCH -p cpu                       # submit to partition cpu
#SBATCH -C ib                        # require inifiniband connectivity

echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load apptainer/latest
module load conda/latest             # need this to use conda commands
conda activate ompi                  # environment with OpenMPI and Python
export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
mpirun --display bindings \
   apptaner exec $SIFS/dem21.sif python ../../mx2.py mx2mod
