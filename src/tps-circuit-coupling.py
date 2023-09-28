#!/usr/bin/env python3
import sys
import os
import numpy as np

from mpi4py import MPI

class CircuitMockSolver:
    def __init__(self):
        pass

    def fetch(self, interface):
        pass

    def solve(self):
        print("And now we're solving the circuit model!", flush=True)
        pass

    def push(self, interface):
        pass

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
import libtps

comm = MPI.COMM_WORLD
rank0 = (comm.Get_rank() == 0)

# TPS solver
tps = libtps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()

# TODO(trevilo): Add assert that solver type supports circuit coupling
# (i.e., that it is cycle-averaged Joule heating)

tps.initialize()

circuit = CircuitMockSolver()

# Instantiate circuit interface
interface = libtps.Tps2Circuit()
interface.Rplasma = 0.0
interface.Lplasma = 0.0
interface.Pplasma = 0.0

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
solve_circuit_every_n = tps.getRequiredInput("cycle-avg-joule-coupled/solve-circuit-every-n")

if rank0:
    print("Maximum number of time steps: ", max_iters)
    print("Solve circuit every n steps: ", solve_circuit_every_n)

tps.solveBegin()

# Run the time loop
while it < max_iters:
    if np.mod(it, solve_circuit_every_n) == 0:
        tps.pushCircuit(interface)
        if rank0:
            print("Iteration = {0:d}".format(it))
            print("Circuit parameters: Rplasma = {0:.3e}, Lplasma = {1:.3e}".format(interface.Rplasma, interface.Lplasma))
        circuit.fetch(interface)
        circuit.solve()
        circuit.push(interface)
        tps.fetchCircuit(interface)

    tps.solveStep()


    it = it+1

tps.solveEnd()
sys.exit (tps.getStatus())
