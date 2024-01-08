#!/usr/bin/env python3
import sys
import os
import numpy as np

from mpi4py import MPI

class ArrheniusSolver:
    def __init__(self):
        self.UNIVERSALGASCONSTANT = 8.3144598;  # J * mol^(-1) * K^(-1)
        self.species_densities = None
        self.efield = None
        self.heavy_temperature = None
        self.reaction_rates = [None, None]
        #Reaction 1: 'Ar + E => Ar.+1 + 2 E', 
        #Reaction 2: 'Ar.+1 + 2 E => Ar + E'
        self.A = [74072.331348, 5.66683445516e-20]
        self.b = [1.511, 0.368]
        self.E = [1176329.772504, -377725.908714] # [J/mol]

    def fetch(self, interface):
        n_reactions =interface.nComponents(libtps.t2bIndex.ReactionRates)
        for r in range(n_reactions):
            print("Reaction ", r+1, ": ", interface.getReactionEquation(r))
        self.species_densities = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False)
        self.efield = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False)
        self.heavy_temperature = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)

        efieldAngularFreq = interface.EfieldAngularFreq()
        print("Electric field angular frequency: ", efieldAngularFreq)



    def solve(self):
        #A_ * pow(temp, b_) * exp(-E_ / UNIVERSALGASCONSTANT / temp);
        self.reaction_rates = [A * np.power(self.heavy_temperature, b) * 
                               np.exp(-E/(self.UNIVERSALGASCONSTANT * self.heavy_temperature))
                               for A,b,E in zip(self.A, self.b, self.E) ]

    def push(self, interface):
        n_reactions =interface.nComponents(libtps.t2bIndex.ReactionRates)
        if n_reactions >= 2:
            rates =  np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False)
            rates[0:self.heavy_temperature.shape[0]] = self.reaction_rates[0]
            rates[self.heavy_temperature.shape[0]:] = self.reaction_rates[1]



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

boltzmann = ArrheniusSolver()

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
    interface.saveDataCollection(cycle=it, time=it)
    tps.fetch(interface)
    
    it = it+1
    print("it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())
