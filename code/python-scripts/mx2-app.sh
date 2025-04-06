#!/bin/bash
# cc-expts/mx2-app.sh 4/6/25 D.C.
# Shell script to run granular-memory simulation program mx2.py containerized
# on a PC, as an example for "My cheat sheet for MPI, GPU, Apptainer, and HPC".
#
# Runs mx2.py in grandparent directory in 'mpi' parallel-processing mode.
# Reads default config file mx2.yaml in grandparent directory modified by
# mx2mod.yaml in current directory.
#
# To run on N cores 1st activate environment 'ompi' then do
#
# ./mx2-app.sh N
#
export pproc=mpi
mpirun -n $1 apptainer exec "$SIFS"/dem21.sif python ../../mx2.py mx2mod |& tee output
