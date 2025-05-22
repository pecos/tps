from mpi4py import MPI

def master_print(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)
