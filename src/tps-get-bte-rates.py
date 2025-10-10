import numpy as np
from mpi4py import MPI
import scipy
import scipy.optimize
import scipy.interpolate
import basis
import spec_spherical as sp
import collision_operator_spherical as colOpSp
import collisions 
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
from   bte_0d3v_batched import bte_0d3v_batched
import cupy as cp
import matplotlib.pyplot as plt
import csv
import sys
import scipy.cluster
from itertools import cycle
import cross_section

ev_to_K               = collisions.TEMP_K_1EV
Td_fac                = 1e-21
c_gamma               = np.sqrt(2 * (scipy.constants.elementary_charge / scipy.constants.electron_mass))
eps                   = 1e-15
N_Avo                 = scipy.constants.Avogadro

parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte")
parser.add_argument("-solver_type", "--solver_type"               , help="solver type", type=str, default="steady-state")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="/work2/10565/ashwathsv/frontera/tps-venv/frontera/tps-venv/tps/boltzmann/BESolver/python/lxcat_data/fully_lumped_argon_mechanism_cs.lxcat")
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-atol", "--atol"                             , help="absolute tolerance", type=float, default=1e-10)
parser.add_argument("-rtol", "--rtol"                             , help="relative tolerance", type=float, default=1e-8)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=1000)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV)" , type=float, default=0.5)
parser.add_argument("-n0"    , "--n0"                             , help="heavy density (1/m^3)" , type=float, default=3.22e22)
parser.add_argument("-ev_max", "--ev_max"                         , help="max energy in the v-space grid" , type=float, default=30)
parser.add_argument("-Nr", "--Nr"                                 , help="radial refinement", type=int, default=128)
parser.add_argument("-profile", "--profile"                       , help="profile", type=int, default=0)
parser.add_argument("-warm_up", "--warm_up"                       , help="warm up", type=int, default=5)
parser.add_argument("-runs", "--runs"                             , help="runs "  , type=int, default=10)
parser.add_argument("-n_pts", "--n_pts"                           , help="number of points for batched solver", type=int, default=10)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=int, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-use_gpu", "--use_gpu"                       , help="use gpus for batched solver", type=int, default=1)
parser.add_argument("-cycles", "--cycles"                         , help="number of max cycles to evolve to compute cycle average rates", type=float, default=5)
parser.add_argument("-dt"    , "--dt"                             , help="1/dt number of denotes the number of steps for cycle", type=float, default=5e-3)
parser.add_argument("-Efreq" , "--Efreq"                          , help="electric field frequency Hz", type=float, default=6e6)
parser.add_argument("-input", "--input"                           , help="tps data file", type=str,  default="")
#python3 bte_0d3v_batched_driver.py --threads 1 -out_fname bte_ss -solver_type steady-state -c lxcat_data/eAr_crs.synthetic.3sp2r -sp_order 3 -spline_qpts 5 -atol 1e-10 -rtol 1e-10 -max_iter 300 -Te 3 -n0 3.22e22 -ev_max 30 -Nr 127 -n_pts 1 -ee_collisions 1 -cycles 2 -dt 1e-3
args                  = parser.parse_args()

def bte_from_tps(Tarr, narr, Er, Ei, collisions_file, solver_type, ee_collisions):
    # INPUTS:
    # Tarr : Array of heavies temperature (1D array of length sDofInt)
    # narr : Array of species number densities (1D array of length sDofInt * nspecies)
    # Er   : Array of Real part of Efield magnitude (1D array of length sDofInt)
    # Ei   : Array of Imaginary part of Efield magnitude (1D array of length sDofInt)
    # nspecies : Number of species passed to BTE = int( len(narr) / len(Tarr) )  

    # Set up the MPI communicators
    comm = MPI.COMM_WORLD
    rank_ = comm.Get_rank()
    size_ = comm.Get_size()

    # First, get the number of species passed from TPS
    nSprem = len(narr) % len(Tarr)
    if nSprem != 0:
        print("Length of species vector not divisible by length of temperature vector. Remainder = ", nSprem, "Aborting...\n")
        sys.stderr.flush()
        comm.Abort(1)
    nSpecies = int(len(narr) / len(Tarr))

    nActSpecies = nSpecies - 2
    all_species           = cross_section.read_available_species(collisions_file)

    args.solver_type = solver_type
    args.ee_collisions = ee_collisions

    if rank_ == 0:
        print("BTE solver_type = ", args.solver_type, ", ee_collisions = ", args.ee_collisions, ", atol = ", args.atol, ", rtol = ", args.rtol, ", Nr = ", args.Nr, ", max-iter = ", args.max_iter)

    # The species indices are based on the TPS ordering
    NEUIDX = nSpecies - 1
    ELEIDX = nSpecies - 2
    IONIDX = 1 # Index of Ar+ in TPS
    EXCIDX = 0 # Index of lumped excited state Ar* in TPS

    # SETUP THE INPUT ARRAYS CONTAINING Temperature, number densities, electric field
    Te = Tarr 
    Tg = Tarr 
    n_pts = len(Te)
    args.n_pts = n_pts

    args.collisions = collisions_file # collision cross-section file
    n0 = np.zeros(n_pts) # n0 = \sum_{sp} (n_{sp}) - n_e
    for i in range(nActSpecies):
        n0 = n0 + narr[i*n_pts:i*n_pts+n_pts]
    
    # ADD THE NEUTRAL NUMBER DENSITY TO n0
    n0 = n0 + narr[NEUIDX*n_pts:NEUIDX*n_pts+n_pts] + eps

    ne = narr[ELEIDX*n_pts:ELEIDX*n_pts+n_pts] + eps # electron number density
    ni = narr[IONIDX*n_pts:IONIDX*n_pts+n_pts] + eps # ion number density

    print()

    ion_deg = (ne / n0) + eps

    ns_by_n0 = np.zeros((len(all_species), n_pts)) # ns_by_n0 defined for all background species with which electrons collide (ie Ar, Ar*)
    # In the collisions file, Ar is index 0, Ar* is index 1
    ns_by_n0[0,0:n_pts] = narr[NEUIDX*n_pts:NEUIDX*n_pts+n_pts] / n0
    ns_by_n0[1,0:n_pts] = narr[EXCIDX*n_pts:EXCIDX*n_pts+n_pts] / n0

    # Getting the electric field magnitude from real and imaginary components
    Emag = np.sqrt((Er+eps)**2 + (Ei+eps)**2)
    EbyN = Emag/n0/Td_fac

    # Definiting mean parameters for temperature and thermal velocity
    Te_mean       = np.mean(Te/ev_to_K)
    args.Te       = Te_mean
    vth           = np.sqrt(Te_mean) * c_gamma
    args.ev_max   = (6 * vth / c_gamma)**2

    ef            = EbyN * n0 * Td_fac

    # Setup grids for BTE solve
    n_grids       = 1
    grid_idx      = 0

    Te          = np.ones(n_grids) * args.Te 
    ev_max      = np.ones(n_grids) * args.ev_max     
    lm_modes    = [[[l,0] for l in range(args.l_max+1)] for i in range(n_grids)]
    nr          = np.ones(n_grids, dtype=np.int32) * args.Nr

    bte_solver  = bte_0d3v_batched(args,ev_max, Te, nr, lm_modes, n_grids, [args.collisions])

    bte_solver.assemble_operators(grid_idx)

    f0         = bte_solver.initialize(grid_idx, n_pts,"maxwellian")
    bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , n0)
    bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , ne)
    bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , ns_by_n0)
    bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , Tg)   
    bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , Er + eps)
    bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , Ei + eps)
    bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
    bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , Emag)
        
    collision_names = bte_solver.get_collision_names()

    if args.use_gpu==1:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        dev_id   = rank_ % num_gpus
        bte_solver.host_to_device_setup(dev_id, grid_idx)
        gpu_device = cp.cuda.Device(dev_id)
        gpu_device.use()

    f0       = bte_solver.get_boltzmann_parameter(grid_idx,"f0")
    ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type)

    if args.use_gpu==1:
        bte_solver.device_to_host_setup(dev_id, grid_idx)
    
    collision_names = bte_solver.get_collision_names()

    if args.use_gpu==1:
        rr_cpu = cp.asnumpy(qoi["rates"])
    else:
        rr_cpu = qoi["rates"]

    # FROM PYTHON, GET THE GLOBAL MAX AND MIN OF THE RATE COEFFICIENTS FOR EACH REACTION

    rates = (rr_cpu * N_Avo)
    for i in range(qoi["rates"].shape[0]):
        rateloc = rates[i,:]
        locmin = np.amin(rateloc)
        locmax = np.amax(rateloc)

        locnumneg = np.sum(rateloc < -1e5)

        glomin = comm.allreduce(locmin, op=MPI.MIN)
        glomax = comm.allreduce(locmax, op=MPI.MAX)
        glonumneg = comm.allreduce(locnumneg, op=MPI.SUM)

        # if i == qoi["rates"].shape[0]-1:
            # lmi = np.argmin(rateloc)
            # print("Rank = ", rank_, "reaction ", i, ", min = ", locmin, ", ind = ", lmi)
            # Rank  Tg  Te n0  ne  nAr_n0  nArs_n0  Er  Ei  Em  rr0  rr1  rr2  rr3  rr4
            # filestr = "%d  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e\n" %(rank_, Tg[lmi], Te, n0[lmi], ne[lmi], ns_by_n0[0,lmi], ns_by_n0[1,lmi], Er[lmi], Ei[lmi], Emag[lmi], rates[0,lmi], rates[1,lmi], rates[2,lmi], rates[3,lmi], rates[4,lmi])
            # print("Rank = ", filestr)
            # with open("plasma_params_Nr120.txt", "a") as file:
            #     file.write(filestr)
        if rank_ == 0 and i == qoi["rates"].shape[0]-1:
            print("Global reaction ", i, ", min = ", glomin, ", max = ", glomax, ", num_negatives = ", glonumneg)
            # sampind = np.argmax(rateloc)
            # print("[Py] Index = ", sampind, "Sample rates = ", rates[:,sampind])
    
    rates[rates < 0.0] = 0.0
    data = rates[1:rates.shape[0]].flatten()


    # if rank_ == 0:
    #     print("Rank = ", rank_, ", BTE solve complete...")

    return data