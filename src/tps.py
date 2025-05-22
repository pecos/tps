#!/usr/bin/env python3
import os
import sys
from time import perf_counter as time
from mpi4py import MPI

# set path to pyTPS library
path = os.path.dirname( os.path.abspath(sys.argv[0]) )
sys.path.append(os.path.join(path, ".libs")
import libtps


comm = MPI.COMM_WORLD
# TPS solver
tps = libtps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()
tps.solve()



sys.exit (tps.getStatus())
