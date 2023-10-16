#!/usr/bin/env python3
import sys
import os
import numpy as np

from mpi4py import MPI

class BoltzmannMockSolver:
    def __init__(self):
        pass

    def fetch(self, interface):
        species_densities = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False)
        efield = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False)
        heavy_temperature = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)

        print("|| species_densities ||_2 = ", np.linalg.norm(species_densities) )
        print("|| efield ||_2 = ", np.linalg.norm(efield) )
        print("||heavy_temperature||_2 = ", np.linalg.norm(heavy_temperature) )

    def solve(self):
        pass

    def push(self, interface):
        electron_temperature =  np.array(interface.HostWrite(libtps.t2bIndex.ElectronTemperature), copy=False)
        electron_temperature[:] = 1.



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
tps.chooseSolver()
tps.initialize()

boltzmann = BoltzmannMockSolver()

interface = libtps.Tps2Boltzmann(tps)
tps.initInterface(interface)

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
print("Max Iters: ", max_iters)
tps.solveBegin()

while it < max_iters:
    tps.solveStep()
    tps.push(interface)
    boltzmann.fetch(interface)
    boltzmann.solve()
    boltzmann.push(interface)
    tps.fetch(interface)
    
    it = it+1
    print("it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())
