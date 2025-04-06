#!/bin/bash
# osubw-app.sh 4/6/25 D.C.
# Two-task sbatch script uses an Apptainer container m4p.sif that
# has OpenMPI to run osu_bw.py, which measures the commnunication
# speed between two MPI ranks. Activates the Conda environment ompi to
# make OpenMPI available outside the container. Must set SIFS to directory
# containing m4p.sif before running script in a directory containing osu_bw.py
#SBATCH -N 2                       # allocate two nodes
#SBATCH -n 4                       # allocate for up to 4 MPI ranks
#SBATCH -p cpu                     # submit to partition cpu
#SBATCH -C ib                      # require inifiniband connectivity
echo nodelist=$SLURM_JOB_NODELIST  # print list of nodes used
module purge                       # unload all modules
module load apptainer/latest
module load conda/latest
conda activate ompi
mpirun -n 2 --display bindings apptainer exec $SIFS/m4p.sif python osu_bw.py

# Alternative mpirun command has '--map-by node' to make run on two nodes.
#mpirun -n 2 --display bindings --map-by node \
#    apptainer exec $SIFS/ompi5.sif python osu_bw.py
