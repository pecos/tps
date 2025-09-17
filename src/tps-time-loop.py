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
tps = pytps.libtps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()

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
