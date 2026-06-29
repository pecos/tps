#!/usr/bin/env python3
import sys
import os
import numpy as np

import configparser

from mpi4py import MPI

# set path to pyTPS library
path = os.path.dirname( os.path.abspath(sys.argv[0]) )
print(path)
sys.path.append(path)
import pytps


comm = MPI.COMM_WORLD
# TPS solver
tps = pytps.libtps.Tps(comm) # Python object which binds to TPS class

tps.parseCommandLineArgs(sys.argv) # binding which calls Tps::parseCommandLineArgs()
tps.parseInput() # calls Tps::parseInput()
tps.chooseDevices() # calls Tps::chooseDevices() (CPU or GPU)
tps.chooseSolver() # calls Tps::chooseSolver() (type of solver needed, we use cycle-avg-joule-coupled for Boltzmann)
tps.initialize()   #initialize the requested solver (Tps::Solver->initialize())

ini_name = pytps.resolve_runFile(sys.argv)
print(ini_name)
config = configparser.ConfigParser()
config.read(ini_name)

boltzmann = pytps.TabulatedSolver(comm, config)

interface = pytps.libtps.Tps2Boltzmann(tps)
tps.initInterface(interface)

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
pytps.master_print(comm,"Max Iters: ", max_iters)
tps.solveBegin()

while it < max_iters:
    tps.push(interface)
    boltzmann.fetch(interface)
    boltzmann.solve()
    boltzmann.push(interface)
    interface.saveDataCollection(cycle=it, time=it)
    tps.fetch(interface)
    tps.solveStep() 

    it = it+1
    pytps.master_print(comm, "it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())

    
    # calls solver->solveStep() (need to split this into flow->step, thermochem->step first)
    # Can further split into reaction-> step so that Python controls TPS calls in a fine-grained manner

    # wtind = 1; wt = 0.01

    # tps.push(interface)
    # boltzmann.fetch(interface)
    # boltzmann.solve_weighted(1,1.0)

    # boltzmann2.fetch(interface)
    # boltzmann2.solve_weighted(wtind,wt)

    # THE RATES COMING FROM boltzmann ARE THE INITIAL RATES (TABULATED)
    # THE RATES COMING FROM boltzmann2 ARE THE NEW RATES (CAN BE FROM BTE)
    # We need to ensure a smooth transition from boltzmann to boltzmann2 without large velocitiy transients
    # So the rates passed to TPS are a linear combination (i.e.) rate = (1-alpha)*boltzmann.rates + alpha*boltzmann2.rates
    # alpha needs to be small to start with and increase with time, approaching to 1
    # We anticipate that this will give a smooth transition from a tabulated to BTE chemistry
    # alpha = np.amin([0.999999,5e-3*it])
    # alpha = 1.0  -> rate coming only from boltzmann2, alpha = 0.0 -> rate solely from boltzmann
    # alpha = 1.0
    # boltzmann.blend_and_push(interface, boltzmann.rates, boltzmann2.rates, alpha)
    # interface.saveDataCollection(cycle=it, time=it)
    # tps.fetch(interface)
    # tps.solveStep()
