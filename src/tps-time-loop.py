#!/usr/bin/env python3
import sys
import os
import numpy as np
import h5py
import scipy
import scipy.interpolate

import configparser

from mpi4py import MPI

def master_print(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)

class NullSolver:
    def __init__(self, comm):
        self.comm = comm
        self.species_densities = None
        self.efield = None
        self.heavy_temperature = None
        #Reaction 1: 'Ar + E => Ar.+1 + 2 E', 
        #Reaction 2: 'Ar.+1 + 2 E => Ar + E'


    def fetch(self, interface):
        n_reactions =interface.nComponents(libtps.t2bIndex.ReactionRates)
        for r in range(n_reactions):
            master_print(self.comm, "Reaction ", r+1, ": ", interface.getReactionEquation(r))
        self.species_densities = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False)
        self.efield = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False)
        self.heavy_temperature = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)

        efieldAngularFreq = interface.EfieldAngularFreq()
        master_print(self.comm,"Species densities:", self.species_densities.min(), " ", self.species_densities.max())
        master_print(self.comm,"Heavy Temp:", self.heavy_temperature.min(), " ", self.heavy_temperature.max())
        master_print(self.comm,"Efield:", self.efield.min(), " ", self.efield.max())
        master_print(self.comm,"Electric field angular frequency: ", efieldAngularFreq)



    def solve(self):
        pass

    def push(self, interface):
        rates =  np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False)

        n_reactions =interface.nComponents(libtps.t2bIndex.ReactionRates)
        for r in range(n_reactions):
            rates[r*self.heavy_temperature.shape[0]:(r+1)*self.heavy_temperature.shape[0]] = (10.**(-r))*1e-6*self.heavy_temperature


class TabulatedSolver:
    def __init__(self, comm, config):
        self.comm = comm
        self.config = config
        self.species_densities = None
        self.efield = None
        self.heavy_temperature = None

        self.tables = self._read_tables()
        self.rates = []

    def _findPythonReactions(self):
        filenames = []
        nreactions = self.config.getint("reactions","number_of_reactions",fallback=0)
        for ir in range(nreactions):
            sublist = self.config["reactions/reaction{0:d}".format(ir+1)]
            rtype = sublist["model"]
            if rtype == "bte":
                filenames.append(sublist["tabulated/filename"].strip("'"))

        return filenames

    def _read_tables(self):
        filenames = self._findPythonReactions()
        
        #["./rate-coefficients/Ionization_Ground.h5",
        #         "./rate-coefficients/Ionization_Lumped.h5",
        #         "./rate-coefficients/Excitation_Lumped.h5"]
        tables = []
        for filename in filenames:
            with h5py.File(filename, 'r') as fid:
                Tcoeff = fid['table'][:]

            tables.append(scipy.interpolate.interp1d(Tcoeff[:,0], Tcoeff[:,1], kind='linear',
                                                     bounds_error=False, fill_value='extrapolate'))

        return tables
    
    def fetch(self, interface):
        n_reactions =interface.nComponents(libtps.t2bIndex.ReactionRates)
        for r in range(n_reactions):
            master_print(self.comm,"Reaction ", r+1, ": ", interface.getReactionEquation(r))
        self.species_densities = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False)
        self.efield = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False)
        self.heavy_temperature = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)

        efieldAngularFreq = interface.EfieldAngularFreq()
        master_print(self.comm,"Electric field angular frequency: ", efieldAngularFreq)

    def solve(self):
        self.rates = []
        for table in self.tables:
            self.rates.append(table(self.heavy_temperature))

    def push(self, interface):
        rates =  np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False)
        offset = 0
        for rate in self.rates:
            rates[offset:offset+rate.shape[0]] = rate
            offset = offset+rate.shape[0]



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

ini_name = ''
if '-run' in sys.argv:
    ini_name = sys.argv[sys.argv.index('-run') + 1 ]
elif '-runFile' in sys.argv:
    ini_name = sys.argv[sys.argv.index('-runFile') + 1 ]
else:
    exit(-1)

config = configparser.ConfigParser()
config.read(ini_name)

boltzmann = TabulatedSolver(comm, config)

interface = libtps.Tps2Boltzmann(tps)
tps.initInterface(interface)

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
master_print(comm,"Max Iters: ", max_iters)
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
    master_print(comm, "it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())
