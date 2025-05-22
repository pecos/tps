#!/usr/bin/env python3
import pathlib
import sys
from time import perf_counter as time
from mpi4py import MPI

# set path to pyTPS library
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(path)
import pytps

comm = MPI.COMM_WORLD
# TPS solver
tps = pytps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()
tps.solve()



sys.exit (tps.getStatus())
