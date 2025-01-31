"""count_mpi.py
Simple time-consuming program requiring minimal memory access, counts an
counts int up to chosen value.

This version uses MPI to run simultaneously on more than one core.

Usage:
    $ mpirun -n <cores> python countdown.py <n>
runs <cores> separate tasks, each of which counts up to <n> then exits.

"""
THIS_IS = 'count_mpi.py 1/31/25 D.C.'

import sys,time
from mpi4py import MPI

# Get and print info on this MPI rank.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
pname = MPI.Get_processor_name()
version = MPI.Get_library_version()
# Truncate version to first comma or newline, otherwise is a long string
version = version.partition(',')[0]
version = version.partition('\n')[0]
print(f'This is rank {rank} of {size}'
      f' on {pname} running {version}')
n = int(sys.argv[1])
print(f'(rank {rank} Counting up to {n:,}...')
t0 = time.perf_counter()
for i in range(n):
    pass
t = time.perf_counter() - t0
print(f'(rank {rank})...done, took {t:.3e}s, {n/t:.3e}counts/s')