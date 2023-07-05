#!/usr/bin/env python3
import sys
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
import libtps as tps

# TPS solver
tps = tps.Tps()

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.solve()

sys.exit (tps.getStatus())
