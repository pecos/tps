import numpy as np
from mpi4py import MPI
import scipy
import scipy.optimize
import scipy.interpolate
import basis
import spec_spherical as sp
import collision_operator_spherical as colOpSp
# import collisions 
import parameters as params
from   time import perf_counter as time, sleep
import utils as BEUtils
import argparse
import scipy.integrate
from   scipy.integrate import ode
from   advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
from   scipy.interpolate import interp1d
import scipy.sparse.linalg
from   datetime import datetime
# from   bte_0d3v_batched import bte_0d3v_batched
import cupy as cp
import matplotlib.pyplot as plt
import csv
import sys
import scipy.cluster
from itertools import cycle
import cross_section

def bte_from_tps(Tarr, narr, Er, Ei, nspecies):
    comm = MPI.COMM_WORLD
    rank_ = comm.Get_rank()
    size_ = comm.Get_size()

    # INPUTS:
    # Tarr : Array of heavies temperature (1D array of length sDofInt)
    # narr : Array of species number densities (1D array of length sDofInt * nspecies)
    # Er   : Array of Real part of Efield magnitude (1D array of length sDofInt)
    # Ei   : Array of Imaginary part of Efield magnitude (1D array of length sDofInt)
    Tarr2 = Tarr
    narr = narr.reshape((nspecies, len(Tarr)))

    if rank_ == 0:
        module_name = 'cross_section'  # Example: checking for the 'os' module
        if module_name in sys.modules:
            print(f"The '{module_name}' module is imported.")
        else:
            print(f"The '{module_name}' module is not imported.")

    

    return Tarr2