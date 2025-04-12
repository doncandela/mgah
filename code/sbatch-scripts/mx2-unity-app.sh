#!/bin/bash
# cc-expts-unity-app/mx2-unity-app.sh 4/12/25 D.C.
# sbatch script to run granular-memory simulation program mx2.py containerized on the
# Unity cluster, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
# Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
# Reads default config file mx2.yaml in grandparent directory modified by
# mx2mod.yaml in current directory.
#SBATCH -n 15                        # run on 15 cores
#SBATCH -N 1                         # use 1 node
# #SBATCH --exclusive                  # don't share nodes with other jobs
#SBATCH --mem=100G                   # allocate 100GB memory per node
# #SBATCH --mem=0                      # allocate all available memory on nodes used
#SBATCH -t 10:00:00                  # time limit 10 hrs (default is 1 hr)
#SBATCH -p cpu                       # submit to partition cpu
# #SBATCH -p cpu,cpu-preempt           # submit to partition cpu or cpu-preempt (<2 hrs)
#SBATCH -C ib                        # require inifiniband connectivity
scontrol write batch_script $SLURM_JOB_ID -;echo # print this script to output
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                         # unload all modules
module load apptainer/latest
module load conda/latest             # need this to use conda commands
conda activate ompi                  # environment with OpenMPI, then don't need OpenMPI module
# module load openmpi/5.0.3            # load OpenMPI module, then don't need ompi environment 
export pproc=mpi                     # tells dem21 to run in MPI-parallel mode
SIFS='/work/pi_candela_umass_edu/dcstuff/sifs'   # where dem21.sif is kept
mpirun --display bindings apptainer exec $SIFS/dem21.sif \
     python ../../mx2.py mx2mod       # run displaying bindings
# mpirun apptainer exec $SIFS/dem21.sif \
#    python ../../mx2.py mx2mod       # run without displaying bindings
