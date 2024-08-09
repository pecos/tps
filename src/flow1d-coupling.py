#!/usr/bin/env python3
import sys
import os
import numpy as np
from mpi4py import MPI
# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
import libtps

comm = MPI.COMM_WORLD
# TPS solver
tps = libtps.Tps(comm)
tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()

print('-'*40, '\nCreating 1d interface instance\n', '-'*40)
interface = libtps.Qms2Flow1d(tps)
print('-'*40, '\nInitializing 1d interface vector size\n', '-'*40)
n_points = 25
interface.initialize(n_points)
interface.print_all()

print('-'*40, '\nCreating NumPy array interfaces for vectors\n', '-'*40)
cond_1d = np.array(interface.PlasmaConductivity1d(), copy = False)
joule_1d = np.array(interface.JouleHeating1d(), copy = False)
radius_1d = np.array(interface.TorchRadius1d(), copy = False)
z_1d = np.array(interface.Coordinates1d(), copy = False)

R_TORCH = 2.75e-1
print('-'*40, '\nSetting vector values\n', '-'*40)
z_1d[0:n_points] = np.linspace(0, 0.5, n_points)
cond_1d[0:n_points] = np.ones(n_points)
radius_1d[0:n_points] = R_TORCH*(1 - 0.1*np.exp(-((z_1d - 0.25)/0.05)**2))
interface.print_all()

print('-'*40, '\nSolving EM field\n', '-'*40)
interface.solve()
interface.print_all()
