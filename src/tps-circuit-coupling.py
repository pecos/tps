#!/usr/bin/env python3
import sys
import os
import numpy as np

from mpi4py import MPI

class CircuitMockSolver:
    def __init__(self):
        pass

    def fetch(self, interface):
        print("Hello from the circuit model fetch!")

    def solve(self):
        print("And now we're solving the circuit model!")
        pass

    def push(self, interface):
        print("Hello from the circuit model push!")

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
print(path + "/.libs")
sys.path.append(path + "/.libs")
import libtps

comm = MPI.COMM_WORLD
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

# TODO(trevilo): Instantiate circuit interface
#interface = libtps.Tps2Boltzmann(tps)

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
solve_circuit_every_n = tps.getRequiredInput("cycle-avg-joule-coupled/solve-circuit-every-n")

print("Maximum number of time steps: ", max_iters)
print("Solve circuit every n steps: ", solve_circuit_every_n)
tps.solveBegin()

# Run the time loop
while it < max_iters:
    tps.solveStep()

    if np.mod(it, solve_circuit_every_n) == 0:
        tps.pushCircuit(interface)
        circuit.fetch(interface)
        circuit.solve()
        circuit.push(interface)
        tps.fetchCircuit(interface)

    it = it+1
    print("it, ", it)

tps.solveEnd()
sys.exit (tps.getStatus())
