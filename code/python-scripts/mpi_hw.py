"""mpi_hw.py 8/7/22 D.C.
Tests if MPI is installed and functional by having each rank print
Hello world plus other info.
"""
from mpi4py import MPI

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
