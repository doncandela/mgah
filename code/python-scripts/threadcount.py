"""
threadcount.py uses timing to estimate the number of threads being used 
while Numpy makes and multipies two large matrices.
"""
THIS_IS = 'threadcount.py 1/28/25 D.C.'
T = 3
M = 1_000    # will multiply two MxM matrices T times
M = 3_000
M = 10_000

SEED = 2025  # seed for making random matrices
#SEED = None  # use unseeded randoms

import numpy as np
import time

rng = np.random.default_rng(SEED)   # get random number generator

print(f'Making {M:,} x {M:,} random matrices...')
tw0 = time.perf_counter()           # wall time
tp0 = time.process_time()           # sum of run times in each core
aa = rng.normal(size=(M,M))
bb = rng.normal(size=(M,M))
cc = np.zeros((M,M))
tw = time.perf_counter() - tw0
tp = time.process_time() - tp0
print(f'...took {tw:.3e}s, average threads = {tp/tw:.3f}')

print(f'Multiplying matrices {T} times...')
tw0 = time.perf_counter()
tp0 = time.process_time()
for t in range(T):
    np.dot(aa,bb,out=cc)
twt = (time.perf_counter() - tw0)/T
tpt = (time.process_time() - tp0)/T
print(f'...took {twt:.3e}s per trial, average threads = {tpt/twt:.3f}')


