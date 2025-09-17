from mpi4py import MPI
import os
import sys

def master_print(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)

def resolve_runFile(cl_args):
    ini_name = ''
    if '-run' in cl_args:
        ini_name = cl_args[cl_args.index('-run') + 1 ]
    elif '--runFile' in sys.argv:
        ini_name = cl_args[cl_args.index('--runFile') + 1 ]
    else:
        print("Could not parse command line in python. GOOD BYE!")
        exit(-1)
    return ini_name


base = os.path.abspath(os.path.dirname( sys.argv[0] ) )
print(os.path.join(base, ".libs"))
sys.path.append(os.path.join(base, ".libs"))
import libtps
