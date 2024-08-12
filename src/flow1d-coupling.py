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

# 1d flow - 2d EM interface
interface = libtps.Qms2Flow1d(tps)
# Initialize MFEM vectors with number of 1d points
n_points = 1000
interface.initialize(n_points)

# Access vectors through NumPy
cond_1d = np.array(interface.PlasmaConductivity1d(), copy = False)
joule_1d = np.array(interface.JouleHeating1d(), copy = False)
radius_1d = np.array(interface.TorchRadius1d(), copy = False)
z_1d = np.array(interface.Coordinates1d(), copy = False)

# Example vector data
R_TORCH = 2.75e-2
z_1d[0:n_points] = np.linspace(0, 0.315, n_points)
cond_1d[0:n_points] = np.ones(n_points)
radius_1d[0:n_points] = R_TORCH

# Solve
interface.set_n_interp(100)
interface.solve()

# Evaluate Joule heating
tot_1d = interface.total_joule_1d()/1e3
tot_2d = interface.total_joule_2d()/1e3
error = 100*np.abs(tot_1d - tot_2d)/np.abs(tot_2d)

print(f'Total 2d Joule heating [kW]: {tot_2d:.2f}')
print(f'Total 1d Joule heating [kW]: {tot_1d:.2f}')
print(f'Error [%]: {error:.3e}')
