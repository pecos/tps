from mpi4py import MPI
import pathlib
import sys

def master_print(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)

path = pathlib.Path(__file__).parent.resolve()
sys.path.append(path + "../.libs")
import libtps
