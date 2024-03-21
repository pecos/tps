#!/usr/bin/env python3
import sys
import os
import cupy as cp
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

#cp.profiler.start()
cp.cuda.nvtx.RangePush("tpsStep")
tps.solveStep()
cp.cuda.nvtx.RangePop()
#tps.solve()

sys.exit (tps.getStatus())
