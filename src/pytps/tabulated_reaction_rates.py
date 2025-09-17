import h5py
import scipy
import scipy.interpolate
import numpy as np
from .utils import master_print, libtps


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
