"""count.py
Simple time-consuming program requiring minimal memory access, counts an
counts int up to chosen value.

Usage:
    $ python countdown.py <n>
counts up to n then exits.

"""
THIS_IS = 'count.py 1/30/25 D.C.'

import sys,time

n = int(sys.argv[1])
print(f'Counting up to {n:,}...')
t0 = time.perf_counter()
for i in range(n):
    pass
t = time.perf_counter() - t0
print(f'...done, took {t:.3e}s, {n/t:.3e}counts/s')