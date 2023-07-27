#!/usr/bin/env python3
import sys
import os

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


it = 0
max_iters = tps.getRequiredInput("flow/maxIters")
print("Max Iters: ", max_iters)
tps.solveBegin()

while it < max_iters:
    tps.solveStep()
    it = it+1
    print("it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())
