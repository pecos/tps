#!/usr/bin/env python3
###  The purpose of this driver is to test TPS when running on a subcommunicator
import sys
import os

from mpi4py import MPI

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/../src/.libs")
print(path + "/../src/.libs")
import libtps as tps

comm = MPI.COMM_WORLD
color = comm.Get_rank() % 2
tpsComm = comm.Split(color, comm.Get_rank())

status = 0

if color == 0:
    # TPS solver
    tps = tps.Tps(tpsComm)

    tps.parseCommandLineArgs(sys.argv)
    tps.parseInput()
    tps.chooseDevices()
    tps.chooseSolver()
    tps.solve()

    status = tps.getStatus()

sys.exit (status)
