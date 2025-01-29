"""
threadcount_mpi.py uses timing to estimate the number of threads being used in
each rank of an MPI program, while Numpy makes and multipies two large
matrices.
"""
THIS_IS = 'threadcount_mpi.py 1/29/25 D.C.'
T = 3
M = 300      # will multiply two MxM matrices T times
M = 1_000
M = 3_000
# M = 10_000

SEED = 2025  # seed for making random matrices
#SEED = None  # use unseeded randoms

import numpy as np
import time

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
print(f'Hello world from rank {rank} of {size}'
      f' on {pname} running {version}')

rng = np.random.default_rng(SEED)   # get random number generator
print(f'(rank {rank}) Making {M:,} x {M:,} random matrices...')
tw0 = time.perf_counter()           # wall time
tp0 = time.process_time()           # sum of run times in each core
aa = rng.normal(size=(M,M))
bb = rng.normal(size=(M,M))
cc = np.zeros((M,M))
tw = time.perf_counter() - tw0
tp = time.process_time() - tp0
print(f'(rank {rank}) ...took {tw:.3e}s, average threads = {tp/tw:.3f}')

print(f'(rank {rank}) Multiplying matrices {T:,} times...')
tw0 = time.perf_counter()
tp0 = time.process_time()
for t in range(T):
    np.matmul(aa,bb,out=cc)
twt = (time.perf_counter() - tw0)/T
tpt = (time.process_time() - tp0)/T
print(f'(rank {rank}) ...took {twt:.3e}s per trial,'
      f' average threads = {tpt/twt:.3f}')


