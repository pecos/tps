#!/usr/bin/env python3
import sys
import pathlib
import numpy as np

import configparser

from mpi4py import MPI

# set path to pyTPS library
path = pathlib.Path(__file__).parent.resolve()
sys.path.append(path)
import pytps


comm = MPI.COMM_WORLD
# TPS solver
tps = pytps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()

ini_name = ''
if '-run' in sys.argv:
    ini_name = sys.argv[sys.argv.index('-run') + 1 ]
elif '--runFile' in sys.argv:
    ini_name = sys.argv[sys.argv.index('--runFile') + 1 ]
else:
    print("Could not parse command line in python. GOOD BYE!")
    exit(-1)

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
