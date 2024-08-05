#!/usr/bin/env python3
import sys
import os
import numpy as np
from mpi4py import MPI
# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
print(sys.path)
import libtps

comm = MPI.COMM_WORLD
# TPS solver
tps = libtps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
#tps.chooseSolver()
#tps.initialize()

print('Creating 1d interface instance')
interface = libtps.Qms2Flow1d(tps)
print('Initializing 1d interface vector size')
interface.initialize(3)
interface.print_all()
print('Writing plasma conductivity from Python')
cond_1d = np.array(interface.PlasmaConductivity1d(), copy = False)
cond_1d[0] = 1
cond_1d[1] = 2
cond_1d[2] = np.pi

interface.print_all()
