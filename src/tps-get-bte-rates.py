import numpy as np
from mpi4py import MPI

def bte_from_tps(Tarr, narr, nspecies):
    comm = MPI.COMM_WORLD
    rank_ = comm.Get_rank()
    size_ = comm.Get_size()
    Tarr2 = Tarr
    narr = narr.reshape((nspecies, len(Tarr)))

    return Tarr2