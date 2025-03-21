#!/usr/bin/env python3
import sys
import os
from time import perf_counter as time
from mpi4py import MPI

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
import libtps as tps

comm = MPI.COMM_WORLD
# TPS solver
tps = tps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()
tps.solveStep()
#tps.solve()

for i in range(20):
    comm.Barrier()
    t1 = time()
    tps.solveStep()
    comm.Barrier()
    t2 = time()
    print("step [%04d] solve time (s)= %.4E"%(i, t2-t1))

sys.exit (tps.getStatus())
