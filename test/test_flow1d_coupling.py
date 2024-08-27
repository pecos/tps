#!/usr/bin/env python3
""" This is a driver to test the interface between a Python 1d flow solver
    and the 2d axisymmetric EM solver. The 1d flow solver is emulated by defining
    the plasma conductivity"""
import sys
import os
import numpy as np
import h5py
from mpi4py import MPI
# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/../src/.libs")
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
N_POINTS = 1000
interface.initialize(N_POINTS)

# Access vectors through NumPy
cond_1d = np.array(interface.PlasmaConductivity1d(), copy = False)
joule_1d = np.array(interface.JouleHeating1d(), copy = False)
radius_1d = np.array(interface.TorchRadius1d(), copy = False)
z_1d = np.array(interface.Coordinates1d(), copy = False)

# Check that Joule heating is read only
if 'WRITE_JOULE' in os.environ:
    joule_1d[0] = 1

# Example vector data
R_TORCH = 2.75e-2
L_TORCH = 0.315
z_1d[0:N_POINTS] = np.linspace(0, L_TORCH, N_POINTS)
cond_1d[0:N_POINTS] = np.ones(N_POINTS)
radius_1d[0:N_POINTS] = R_TORCH

# Solve
N_INTERP = 100
interface.set_n_interp(N_INTERP)
interface.solve()

if 'TOTAL_POWER' in os.environ:
    power = interface.total_joule_2d()/1e3
    assert np.abs(power - 20)/20 < 0.05, 'Total power should be 20 kW'

if 'COMPARE_JOULE' in os.environ:
    power_1d = interface.total_joule_1d()/1e3
    power_2d = interface.total_joule_2d()/1e3
    error = np.abs(power_1d - power_2d)/np.abs(power_2d)
    assert error < 1e-5, '1d and 2d Joule heating integrals should match'

SOLN_FILE = "flow1d_coupling.sol.h5"
# Save output with hdf5
with h5py.File(SOLN_FILE, "w") as f:
    _ = f.create_dataset('input/axial_coordinates', data=z_1d)
    _ = f.create_dataset('input/torch_radius', data=radius_1d)
    _ = f.create_dataset('input/plasma_conductivity', data=cond_1d)
    _ = f.create_dataset('output/joule_heating', data=joule_1d)
