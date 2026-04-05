import argon.scripts.synthetic_cs as synthetic_cs
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
import utils as bte_utils
import cupy as cp
import matplotlib.pyplot as plt
import csv
import sys
import scipy.cluster
from itertools import cycle
import cross_section
import os

ev_to_K               = collisions.TEMP_K_1EV
Td_fac                = 1e-21
me                    = scipy.constants.electron_mass
c_gamma               = np.sqrt(2 * (scipy.constants.elementary_charge / me))
eps                   = 1e-15
N_Avo                 = scipy.constants.Avogadro

EMag_threshold        = 1e-10
rand_seed             = 0
clstr_maxiter         = 10
clstr_threshold       = 1e-3
n0_param              = 3.22e22 
kB                    = scipy.constants.Boltzmann

varyT_cs = 0
append_recomb_cs = 1

logfile = "outlog.txt"

parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte")
parser.add_argument("-solver_type", "--solver_type"               , help="solver type", type=str, default="steady-state")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="/work2/10565/ashwathsv/frontera/tps-venv/frontera/tps-venv/tps/boltzmann/BESolver/python/lxcat_data/fully_lumped_argon_mechanism_cs_recomb.lxcat")
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=11)
parser.add_argument("-atol", "--atol"                             , help="absolute tolerance", type=float, default=1e-20)
parser.add_argument("-rtol", "--rtol"                             , help="relative tolerance", type=float, default=1e-10)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=50)
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
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=1)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=int, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-use_gpu", "--use_gpu"                       , help="use gpus for batched solver", type=int, default=1)
parser.add_argument("-cycles", "--cycles"                         , help="number of max cycles to evolve to compute cycle average rates", type=float, default=5)
parser.add_argument("-dt"    , "--dt"                             , help="1/dt number of denotes the number of steps for cycle", type=float, default=1e-3)
parser.add_argument("-Efreq" , "--Efreq"                          , help="electric field frequency Hz", type=float, default=6e6)
parser.add_argument("-input", "--input"                           , help="tps data file", type=str,  default="")
#python3 bte_0d3v_batched_driver.py --threads 1 -out_fname bte_ss -solver_type steady-state -c lxcat_data/eAr_crs.synthetic.3sp2r -sp_order 3 -spline_qpts 5 -atol 1e-10 -rtol 1e-10 -max_iter 300 -Te 3 -n0 3.22e22 -ev_max 30 -Nr 127 -n_pts 1 -ee_collisions 1 -cycles 2 -dt 1e-3
args                  = parser.parse_args()

def bte_from_tps(Tarr, narr, Er, Ei, collisions_file, nBTEreactions, solver_type, ee_collisions, n_grids, grid_idx_to_npts, grid_idx_to_spatial_idx_map_vec, use_interp, n_sub_clusters, Te_vec, Nr, rtol, store_csv, dt_bte):
    # INPUTS:
    # Tarr : Array of heavies temperature (1D array of length sDofInt)
    # narr : Array of species number densities (1D array of length sDofInt * nspecies)
    # Er   : Array of Real part of Efield magnitude (1D array of length sDofInt)
    # Ei   : Array of Imaginary part of Efield magnitude (1D array of length sDofInt)
    # nspecies : Number of species passed to BTE = int( len(narr) / len(Tarr) )  
    # n_grids : number of v-space grids
    # grid_idx_to_npts : vector of length n_grids (ith element gives number of points in ith grid)
    # grid_idx_to_spatial_idx_map_vec : vector of length sDofInt_ (the elements are indices of the points belonging to each grid)
    # For example, for i = 0 grid, grid_idx_to_spatial_idx_map[0:grid_idx_to_npts[0]] gives the indices of the points belonging to the grid

    # Set up the MPI communicators
    comm = MPI.COMM_WORLD
    rank_ = comm.Get_rank()
    size_ = comm.Get_size()

    crs_folder = "lxcat_data/argon/crs_lxcat"
    if rank_ == 0:
        if varyT_cs == 1:
            if not os.path.exists(crs_folder):
                os.makedirs(crs_folder)
    comm.Barrier()

    Te                               = Te_vec
    #  generate crs Tg depended crs data 
    col_cs = list()
    for idx in range(n_grids):
        cs_fname = collisions_file
        if varyT_cs == 1:
            cs_fname = "%s/crs_rank_%06d_npes_%06d_%06d.txt"%(crs_folder, rank_, size_, idx) 
            synthetic_cs.gen_lxcat_file(cs_fname, Te[idx] * ev_to_K, append_recomb_cs, rank_, idx)
        col_cs.append(cs_fname)
    comm.Barrier()

    args.collisions = collisions_file # collision cross-section file
    if varyT_cs == 1:
        args.collisions = col_cs[0]
        if rank_ == 0:
            print("Rank ", rank_, ", args.collisions = ", args.collisions, flush=True)
    cs_data_all   = cross_section.read_cross_section_data(args.collisions)


    args.Nr = Nr
    args.store_csv = store_csv
    args.rtol = rtol
    if solver_type == "transient":
        args.dt = dt_bte

    # Write the grid_idx_to_spatial_pts vector to a list where each element of the list contains the indices for grid_idx
    glo = 0
    grid_idx_to_spatial_idx_map = list()
    for gidx in range(n_grids):
        ghi = glo + grid_idx_to_npts[gidx]
        grid_idx_to_spatial_idx_map.append(grid_idx_to_spatial_idx_map_vec[glo:ghi])
        glo = ghi

    # First, get the number of species passed from TPS
    nSprem = len(narr) % len(Tarr)
    if nSprem != 0:
        print("Length of species vector not divisible by length of temperature vector. Remainder = ", nSprem, "Aborting...\n")
        sys.stderr.flush()
        comm.Abort(1)
    nSpecies = int(len(narr) / len(Tarr))

    nActSpecies = nSpecies - 2
    all_species           = cross_section.read_available_species(args.collisions)

    args.solver_type = solver_type
    args.ee_collisions = ee_collisions

    if rank_ == 0:
        print("[Python] BTE solver_type = ", args.solver_type, ", ee_collisions = ", args.ee_collisions, ", rtol = ", args.rtol, ", Nr = ", args.Nr, ", max-iter = ", args.max_iter, "do_sub_clust = ", use_interp, ", n_sub_clusters = ", n_sub_clusters, ", store_csv = ", store_csv, ", dt_BTE = ", dt_bte)

    # The species indices are based on the TPS ordering
    NEUIDX = nSpecies - 1
    ELEIDX = nSpecies - 2
    IONIDX = 1 # Index of Ar+ in TPS
    EXCIDX = 0 # Index of lumped excited state Ar* in TPS

    # SETUP THE INPUT ARRAYS CONTAINING Temperature, number densities, electric field
    Tg = Tarr 
    n_pts = len(Tg)
    
    collision_count = 0
    for col_str, col_data in cs_data_all.items():
        collision_count+=1

    n0 = np.zeros(n_pts) # n0 = \sum_{sp} (n_{sp}) - n_e
    for i in range(nActSpecies):
        n0 = n0 + narr[i*n_pts:i*n_pts+n_pts]
    
    # ADD THE NEUTRAL NUMBER DENSITY TO n0
    n0 = n0 + narr[NEUIDX*n_pts:NEUIDX*n_pts+n_pts]

    ne = narr[ELEIDX*n_pts:ELEIDX*n_pts+n_pts] # electron number density
    ni = narr[IONIDX*n_pts:IONIDX*n_pts+n_pts] # ion number density

    ns_by_n0 = np.zeros((len(all_species), n_pts)) # ns_by_n0 defined for all background species with which electrons collide (ie Ar, Ar*)
    # In the collisions file, Ar is index 0, Ar* is index 1
    for i in range(len(all_species)):
        if all_species[i] == 'Ar':
            ns_by_n0[i] = narr[NEUIDX*n_pts:NEUIDX*n_pts+n_pts] / n0
        elif all_species[i] == 'Ar*':
            ns_by_n0[i] = narr[EXCIDX*n_pts:EXCIDX*n_pts+n_pts] / n0
        elif all_species[i] == 'Ar+':
            ns_by_n0[i] = (ni / n0)

    # enforce mixture mass is normalized
    ns_by_n0 = ns_by_n0/np.sum(ns_by_n0, axis=0)

    # INITIALIZE THE BOLTZMANN SOLVER OBJECT
    lm_modes                         = [[[l,0] for l in range(args.l_max+1)] for grid_idx in range(n_grids)]
    nr                               = np.ones(n_grids, dtype=np.int32) * args.Nr
    vth                              = np.sqrt(Te) * c_gamma
    ev_max                           = (6 * vth / c_gamma)**2 
    # Replace ev_max based on the mean electron energy in each grid
    for idx in range(n_grids):
        ev_max[idx] = 36 * np.mean( Tg[grid_idx_to_spatial_idx_map[idx]] / ev_to_K) 
        if args.ee_collisions==1:
            ev_max[idx] = 100 * np.mean( Tg[grid_idx_to_spatial_idx_map[idx]] / ev_to_K) 

    #  generate crs Tg depended crs data 
    # col_cs = list()
    # Tlfile = "%s/Tlump.txt" %(crs_folder)
    # Tlarr = np.loadtxt(Tlfile, delimiter='\t', ndmin=1)
    # for idx in range(n_grids):
    #     cs_fname = collisions_file
    #     if varyT_cs == 1:
    #         # Tc = Te[idx] * ev_to_K
    #         # cs_T = Tlarr[np.argmin(np.abs(Tlarr-Tc))]
    #         # cs_fname = "%s/crs_Tlump%06d.lxcat"%(crs_folder, int(cs_T)) 
    #     col_cs.append(cs_fname)

    bte_solver  = bte_0d3v_batched(args, ev_max, Te, nr, lm_modes, n_grids, col_cs)

    # compute BTE operators
    for grid_idx in range(n_grids):
        bte_solver.assemble_operators(grid_idx)

    # Initialize the EEDFs to Maxwellian for each v-space grid
    for grid_idx in range(n_grids):
        assert grid_idx_to_npts[grid_idx] > 0
        if(use_interp == 1):
            # 8D subclustering for each grid
            bte_solver.set_boltzmann_parameter(grid_idx, "f_mw", bte_solver.initialize(grid_idx, n_sub_clusters, "maxwellian"))
        else:
            bte_solver.set_boltzmann_parameter(grid_idx, "f_mw", bte_solver.initialize(grid_idx, grid_idx_to_npts[grid_idx], "maxwellian"))

    active_grid_idx  = [i for i in range(n_grids)]

    # Setting up the Boltzmann solver (solve_init())
    num_gpus = 1
    if args.use_gpu==1:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        xp   = cp
    else:
        xp   = np

    dev_id = rank_ % num_gpus

    for grid_idx in range(n_grids):
        if n_grids == num_gpus:
            dev_id = grid_idx % num_gpus
        def t1(dev_id, bte_solver):
            bte_solver.host_to_device_setup(dev_id, grid_idx)
            
        t1(dev_id, bte_solver)

    def ts_op_setup(grid_idx, xp, use_interp, n_sub_clusters, bte_solver):
        f_mw                                    = bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
        n_pts                                   = f_mw.shape[1]
        Qmat                                    = bte_solver._op_qmat[grid_idx]
        INr                                     = xp.eye(Qmat.shape[1])
        bte_solver._op_imat_vx[grid_idx]        = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr)
            
        if use_interp == 1:
            assert n_pts == n_sub_clusters

    if(args.use_gpu==1):
        for grid_idx in active_grid_idx:
            if n_grids == num_gpus:
                dev_id = grid_idx % num_gpus

            def t1(bte_solver):
                ts_op_setup(grid_idx, cp, use_interp, n_sub_clusters, bte_solver)
                    
                vth          = bte_solver._par_vth[grid_idx]
                qA           = bte_solver._op_diag_dg[grid_idx]
                mw           = bte_utils.get_maxwellian_3d(vth, 1)
                mm_op        = bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                f_mw         = bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                f_mw         = f_mw/cp.dot(mm_op, f_mw)
                f_mw         = cp.dot(qA.T, f_mw)
                    
                bte_solver.set_boltzmann_parameter(grid_idx, "u0", cp.copy(f_mw))
            
            with cp.cuda.Device(dev_id):
                t1(bte_solver)
    else:
        for grid_idx in active_grid_idx:
            def t1(bte_solver):
                ts_op_setup(grid_idx, np, use_interp, n_sub_clusters, bte_solver)
                    
                vth          = bte_solver._par_vth[grid_idx]
                qA           = bte_solver._op_diag_dg[grid_idx]
                mw           = bte_utils.get_maxwellian_3d(vth, 1)
                mm_op        = bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                f_mw         = bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                f_mw         = f_mw/np.dot(mm_op, f_mw)
                f_mw         = np.dot(qA.T, f_mw)
                bte_solver.set_boltzmann_parameter(grid_idx, "u0", np.copy(f_mw))
                
            t1(bte_solver)

    # -------------End of solve_init() ------------------------------

    # 8D SUB-CLUSTERING OF THE PLASMA PARAMETERS
    EMag                    = np.sqrt( Er**2 + Ei**2 )
    e_idx                   = EMag < EMag_threshold

    ExbyN                   = Er/n0/Td_fac
    EybyN                   = Ei/n0/Td_fac

    ExbyN[e_idx]            = (EMag_threshold/np.sqrt(2)) / n0[e_idx] / Td_fac
    EybyN[e_idx]            = (EMag_threshold/np.sqrt(2)) / n0[e_idx] / Td_fac

    Ex                      = ExbyN * n0_param * Td_fac
    Ey                      = EybyN * n0_param * Td_fac

    ion_deg                 = ne/n0
    ion_deg[ion_deg<=0]     = 1e-16
    ns_by_n0[ns_by_n0<=0]   = 0
    m_bte                   = np.concatenate([ExbyN.reshape((-1, 1)), EybyN.reshape((-1, 1)), Tg.reshape((-1, 1)), ion_deg.reshape((-1, 1))] + [ ns_by_n0[i].reshape((-1, 1)) for i in range(ns_by_n0.shape[0])], axis=1)

    sub_cluster_c      = None
    sub_cluster_c_lbl  = None

    gidx_to_pidx            = grid_idx_to_spatial_idx_map

    if (use_interp == 1):
        sub_cluster_c           = [None for i in range(n_grids)]
        sub_cluster_c_lbl       = [None for i in range(n_grids)]

        # print("[Python] Rank [%d/%d]: Entered sub-clustering function"%(rank_, size_), flush=True)
        # comm.Barrier()
            
        def normalize(obs, xp):
            std_obs   = xp.std(obs, axis=0)
            std_obs[std_obs == 0.0] = 1.0
            return obs/std_obs, std_obs

        for grid_idx in active_grid_idx:

            if n_grids == num_gpus:
                dev_id = grid_idx % num_gpus

            def t1(bte_solver):
                # to repoduce clusters
                xp                           = np
                np.random.seed(rand_seed)
                m                            = m_bte[gidx_to_pidx[grid_idx]]
                mw , mw_std                  = normalize(m, xp)

                if mw.shape[0] >= n_sub_clusters:
                    mcw0                         = mw[np.random.choice(mw.shape[0], n_sub_clusters, replace=False)]
                else:
                    mcw0                         = mw[np.random.choice(mw.shape[0], n_sub_clusters, replace=True)]

                # gidx 3, Rank 0 fails in below line (hangs without any error)
                mcw                          = scipy.cluster.vq.kmeans2(mw, mcw0, iter=clstr_maxiter, thresh=clstr_threshold, check_finite=False)[0]

                mcw0[0:mcw.shape[0], :]      = mcw[:,:]
                mcw                          = mcw0
                dist_mat                     = xp.linalg.norm(mw[:, None, :] - mcw[None, : , :], axis=2)
                membership_m                 = xp.argmin(dist_mat, axis=1)

                assert mcw.shape[0]          == n_sub_clusters
                mc                           = mcw * mw_std
                sub_cluster_c[grid_idx]      = mc
                sub_cluster_c_lbl[grid_idx]  = membership_m

                n0       = xp.ones(mc.shape[0]) * n0_param
                Ex       = mc[: , 0] * n0_param * Td_fac
                Ey       = mc[: , 1] * n0_param * Td_fac
                Tg       = mc[: , 2]
                ne       = mc[: , 3] * n0_param
                ni       = mc[: , 3] * n0_param
                ns_by_n0 = xp.transpose(mc[: , 4:])
                EMag     = xp.sqrt(Ex**2 + Ey**2)

                if args.verbose == 1 :
                    print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(rank_, size_, grid_idx), flush=True)
                    if rank_ == 0 and grid_idx == 0:
                        print("Efreq = %.4E [1/s]" %(args.Efreq), flush=True)
                    print("n_pts = %d" % grid_idx_to_npts[grid_idx], flush=True)
                        
                    print("n0    (min)               = %.12E [1/m^3]      \t n0   (max) = %.12E [1/m^3] "%(xp.min(n0)       , xp.max(n0))   , flush=True)
                    print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN)    , xp.max(ExbyN)), flush=True)
                    print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN)    , xp.max(EybyN)), flush=True)
                    print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)       , xp.max(Tg))   , flush=True)
                    print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)       , xp.max(ne))   , flush=True)
                        
                    for i in range(ns_by_n0.shape[0]):
                        print("[%d] ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(i, xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
                
                if (args.use_gpu==1):
                    with cp.cuda.Device(dev_id):
                        n0   = cp.array(n0) 
                        Ex   = cp.array(Ex)
                        Ey   = cp.array(Ey)
                        Tg   = cp.array(Tg)
                        ne   = cp.array(ne)
                        ni   = cp.array(ni)
                        EMag = cp.sqrt(Ex**2 + Ey**2)
                        ns_by_n0 = cp.array(ns_by_n0)
                
                # FIX FOR RECOMBINATION (HARDCODED) (ASSUME AMBIPOLARITY i.e. ne = ni)
                for i in range(len(all_species)):
                    if all_species[i] == 'Ar+':
                        ns_by_n0[i] = (ne/n0)**2 * n0
                bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)    
                bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)

                return
            
            with cp.cuda.Device(dev_id):
                t1(bte_solver)
    else:
        
        for grid_idx in active_grid_idx:

            if n_grids == num_gpus:
                dev_id = grid_idx % num_gpus                
            
            def t1(bte_solver):
                bte_idx           = gidx_to_pidx[grid_idx]
                mc                = xp.array(m_bte[bte_idx])
                    
                n0                = xp.ones(mc.shape[0]) * n0_param
                Ex                = mc[: , 0] * n0_param * Td_fac
                Ey                = mc[: , 1] * n0_param * Td_fac

                Tg                = mc[: , 2]
                ne                = mc[: , 3] * n0_param
                ni                = mc[: , 3] * n0_param
                ns_by_n0          = xp.transpose(mc[: , 4:])
                EMag              = xp.sqrt(Ex**2 + Ey**2)

                if args.verbose == 1 :
                    print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(rank_, size_, grid_idx), flush=True)
                    print("Efreq = %.4E [1/s]" %(args.Efreq)      , flush=True)
                    print("n_pts = %d" % grid_idx_to_npts[grid_idx], flush=True)
                        
                    print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN), xp.max(ExbyN)), flush=True)
                    print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN), xp.max(EybyN)), flush=True)
                    print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)   , xp.max(Tg))   , flush=True)
                    print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)   , xp.max(ne))   , flush=True)
                        
                    for i in range(ns_by_n0.shape[0]):
                        print("ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
                
                if (args.use_gpu == 1):
                    with cp.cuda.Device(dev_id):
                        n0   = cp.array(n0) 
                        ne   = cp.array(ne)
                        ni   = cp.array(ni)
                        Ex   = cp.array(Ex)
                        Ey   = cp.array(Ey)
                        Tg   = cp.array(Tg)
                        EMag = cp.sqrt(Ex**2 + Ey**2)
                        ns_by_n0 = cp.array(ns_by_n0)
                
                # FIX FOR RECOMBINATION (HARDCODED) (ASSUME AMBIPOLARITY i.e. ne = ni)
                for i in range(len(all_species)):
                    if all_species[i] == 'Ar+':
                        ns_by_n0[i] = (ne/n0)**2 * n0
                
                # ns_by_n0 obtained here for Ar+ is actually (ni/n0)**2 * n0. Need to fix this
                bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)
                bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)

                return
            
            with cp.cuda.Device(dev_id):
                t1(bte_solver)


    # Rank 0 does not get here at the 6th iteration!!
    # print("[Python] Rank [%d/%d]: Sub-clustering complete"%(rank_, size_), flush=True)
    # comm.Barrier()

    # Do the Boltzmann solve
    qoi_list                = [None for grid_idx in range(n_grids)]
    ff_list                 = [None for grid_idx in range(n_grids)]

    for grid_idx in active_grid_idx:
        if n_grids == num_gpus:
            dev_id = grid_idx % num_gpus

        def t1(bte_solver, ff_list, qoi_list):
            try:
                f0 = bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                ls_c     = 1e-4
                if args.ee_collisions == 1:
                # disabling ee-collisions temporary by force to get a good initial guess. 
                    args.ee_collisions = 0  
                    ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type,
                                alpha_min = 1e-16, alpha_rho=0.5, line_search_c = ls_c)
                    bte_solver.set_boltzmann_parameter(grid_idx, "u0"       , ff)
                    print("Rank ", rank_, ", gidx ", grid_idx, ", setup initial conditions for ee interactions solve", flush=True)

                # enable ee-collisions
                    args.ee_collisions = 1                
                f0       = bte_solver.get_boltzmann_parameter(grid_idx,"u0")
                ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type,
                            alpha_min = 1e-16, alpha_rho=0.5, line_search_c = ls_c)
                
                ff_list[grid_idx]  = ff
                qoi_list[grid_idx] = qoi
            except Exception as e:
                print(e)
                print("rank [%d/%d] solver failed for v-space grid no %d"%(rank_, size_, grid_idx), flush=True)
                comm.Abort(0)
            
        with cp.cuda.Device(dev_id):
            t1(bte_solver, ff_list, qoi_list)
        
    for grid_idx in active_grid_idx:
        if n_grids == num_gpus:
            dev_id = grid_idx % num_gpus
        def t1(bte_solver, ff_list):
            bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", ff_list[grid_idx])
        t1(bte_solver, ff_list)
    
    # PREPARE THE RATES FOR PUSHING TO TPS
    heavy_temp = Tg
    tps_npts    = len(heavy_temp)
    rates       = np.zeros((collision_count, tps_npts))

    cs_data_all   = cross_section.read_cross_section_data(args.collisions)
    if (use_interp==1):
        if(nBTEreactions>0):
            rates[:,:] = 0.0
            for grid_idx in active_grid_idx:

                if n_grids == num_gpus:
                    dev_id = grid_idx % num_gpus

                def t1(bte_solver, sub_cluster_c_lbl, n_sub_clusters, grid_idx_to_spatial_idx_map, coll_count):
                    qA        = bte_solver._op_diag_dg[grid_idx]
                    u0        = bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                    h_curr    = bte_solver.normalized_distribution(grid_idx, u0)
                    qoi       = bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                    rr_cpu    = xp.asnumpy(qoi["rates"])
                    inp_mask  = xp.asnumpy(sub_cluster_c_lbl[grid_idx]) == np.arange(n_sub_clusters)[:, None]
                    rr_interp = np.zeros((coll_count, len(grid_idx_to_spatial_idx_map[grid_idx])))
                        
                    for c_idx in range(n_sub_clusters):
                        inp_idx = inp_mask[c_idx]
                        for r_idx in range(coll_count):
                            rr_interp[r_idx, inp_idx] = rr_cpu[r_idx][c_idx] * N_Avo

                    for r_idx in range(coll_count):                            
                        rates[r_idx][grid_idx_to_spatial_idx_map[grid_idx]] = rr_interp[r_idx, :]
                    
                with cp.cuda.Device(dev_id):
                    t1(bte_solver, sub_cluster_c_lbl, n_sub_clusters, grid_idx_to_spatial_idx_map, collision_count)

            # rates = rates.reshape((-1))
            # rates[rates<0] = 0.0
    else:
        if(nBTEreactions>0):
            rates[:,:] = 0.0
            for grid_idx in active_grid_idx:

                if n_grids == num_gpus:
                    dev_id = grid_idx % num_gpus

                def t1(bte_solver, grid_idx_to_spatial_idx_map, coll_count):
                    qA       = bte_solver._op_diag_dg[grid_idx]
                    u0       = bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                    h_curr   = bte_solver.normalized_distribution(grid_idx, u0)
                    qoi      = bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                    rr_cpu   = xp.asnumpy(qoi["rates"])
                        
                    for r_idx in range(coll_count):
                        rates[r_idx][grid_idx_to_spatial_idx_map[grid_idx]] = rr_cpu[r_idx] * N_Avo

                with cp.cuda.Device(dev_id):
                    t1(bte_solver, grid_idx_to_spatial_idx_map, collision_count)

    
    # FIX THE RECOMBINATION RATES
    # SINCE THE BTE ASSUMES A PSEUDO 2-BODY COLLISION MODEL,
    # THE RECOMBINATION RATE COEFFICIENTS COMING FROM BTE HAVE UNITS OF m^6-s^{-1}
    # WE NEED TO MULTIPLY THESE BY N_avo**2 TO CONVERT THE UNITS TO m^6-mol^{-2}-s^{-1} FOR USE IN TPS
    col_count = 0
    for col_str, col_data in cs_data_all.items():
        print("col_count = ", col_count, col_str)
        if col_data["type"] == "ATTACHMENT":
            rates[col_count][:] = N_Avo*rates[col_count][:]
        col_count+=1
        

    # STORE THE QoIs TO CSV FILE AND PLOT DATA
    if args.store_csv==1:
        for grid_idx in active_grid_idx:

            if n_grids == num_gpus:
                dev_id = grid_idx % num_gpus
                
            with cp.cuda.Device(dev_id):
                u_vec   = bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                qA      = bte_solver._op_diag_dg[grid_idx]
                h_curr  = xp.dot(qA, u_vec)
                h_curr  = bte_solver.normalized_distribution(grid_idx, h_curr)
                ff      = h_curr
                qoi     = bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)

                coll_list               = bte_solver.get_collision_list()[grid_idx]
                coll_names              = bte_solver.get_collision_names()[grid_idx]
                cs_data                 = bte_solver.get_cross_section_data()[grid_idx]

                cs_species              = list()
                for col_idx, (k,v) in enumerate(cs_data.items()):
                    cs_species.append(v["species"])

                cs_species = list(sorted(set(cs_species), key=cs_species.index))

                def asnumpy(a):
                    if cp.get_array_module(a)==cp:
                        with cp.cuda.Device(dev_id):
                            return cp.asnumpy(a)
                    else:
                        return a

                ff_cpu   = asnumpy(ff)
                ev       = np.linspace(1e-3, bte_solver._par_ev_range[grid_idx][1], 500)
                ff_r     = bte_solver.compute_radial_components(grid_idx, ev, ff_cpu)

                n0       = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "n0"))
                ne       = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "ne"))
                ni       = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "ni"))
                ns_by_n0 = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "ns_by_n0")).T

                for i in range(len(all_species)):
                    if all_species[i] == "Ar+":
                        ns_by_n0[:,i] = ni / n0
                Tg       = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "Tg"))
                eRe      = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "eRe"))
                eIm      = asnumpy(bte_solver.get_boltzmann_parameter(grid_idx, "eIm"))
                eMag     = np.sqrt(eRe**2 + eIm**2)

                data_csv = np.zeros((ne.shape[0], 7 + ns_by_n0.shape[1] + len((coll_list)) + 2))

                data_csv[: , 0]    = n0
                data_csv[: , 1]    = ne/n0
                idx                = 2 + ns_by_n0.shape[1]
                data_csv[: ,2:idx] = ns_by_n0[:,:]
            
                data_csv[: , idx]      = Tg
                data_csv[: , idx+1]    = eRe
                data_csv[: , idx+2]    = eIm
                data_csv[: , idx+3]    = eMag
                data_csv[: , idx+4]    = asnumpy(qoi["energy"])
                data_csv[: , idx+5]    = asnumpy(qoi["mobility"])
                data_csv[: , idx+6]    = asnumpy(qoi["diffusion"])

                for col_idx, g in enumerate(coll_list):
                    data_csv[: , idx + 7 + col_idx]    = asnumpy(qoi["rates"][col_idx])

                fname=args.out_fname+"_grid_%02d_rank_%d_npes_%d"%(grid_idx, rank_, size_)
                with open("%s_qoi.csv"%(fname), 'w', encoding='UTF8') as f:
                    writer = csv.writer(f,delimiter=',')
                    # write the header
                    header = ["n0", "ne/n0"] + ["(%s)/n0"%(s) for s in cs_species] + ["Tg", "eRe", "eIm", "E",  "energy", "mobility", "diffusion"]
                    for col_idx, g in enumerate(coll_list):
                        header.append(str(coll_names[col_idx]))
                
                    writer.writerow(header)
                    writer.writerows(data_csv)

                if args.plot_data==1:
                    n_pts        = ff_cpu.shape[1]
                    num_sh       = len(bte_solver._par_lm[grid_idx])
                    num_subplots = num_sh 
                    num_plt_cols = min(num_sh, 4)
                    num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
                    fig          = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=200, constrained_layout=True)
                    plt_idx      =  1
                    n_pts_step   =  n_pts // 4

                    for lm_idx, lm in enumerate(bte_solver._par_lm[grid_idx]):
                        plt.subplot(num_plt_rows, num_plt_cols, plt_idx)

                        for ii in range(0, n_pts, n_pts_step):
                            fr     = np.abs(ff_r[ii, lm_idx, :])
                            mf_str = " ".join([r"$%s/n0$=%.2E"%(s, ns_by_n0[ii, s_idx]) for s_idx, s in enumerate(cs_species)])
                            plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td] $n_e/n_0$=%.2E "%(Tg[ii], eMag[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]) + " " +mf_str)

                        plt.xlabel(r"energy (eV)")
                        plt.ylabel(r"$f_%d$"%(lm[0]))
                        plt.grid(visible=True)
                        if lm_idx==0:
                            plt.legend(prop={'size': 6})

                        plt_idx +=1
                    
                    plt.savefig("%s_plot.png"%(fname))
                    plt.close()

    # for i in range(rates.shape[0]):
    #     rateloc = rates[i,:]
    #     locmin = np.amin(rateloc)
    #     locmax = np.amax(rateloc)

    #     glomin = comm.allreduce(locmin, op=MPI.MIN)
    #     glomax = comm.allreduce(locmax, op=MPI.MAX)
        
    #     locnumneg = np.sum(rateloc < -1e5)
    #     glonumneg = comm.allreduce(locnumneg, op=MPI.SUM)

    #     if rank_ == 0:
    #         printstr = "[Py] Reaction %d, %.4e to %.4e, numneg = %d" %(i, glomin, glomax, glonumneg)
    #         print(printstr)

    data = rates[1:rates.shape[0]].flatten()
    data[data < 1.0e-21] = 0.0
    comm.Barrier()

    return data

# FUNCTION TO SETUP V-SPACE GRIDS FOR THE BTE SOLVE
def bte_grid_setup(Tarr, n_grids):

    # Set up the MPI communicators
    comm = MPI.COMM_WORLD
    rank_ = comm.Get_rank()
    size_ = comm.Get_size()

    if n_grids == 1:
        Te = np.array([max(0.1,np.mean(Tarr / ev_to_K))])
        grid_idx_to_npts = np.array([len(Tarr)], dtype=np.int32)
        grid_pts_to_spatial_index_map_vec = np.array([i for i in range(len(Tarr))], dtype=np.int64)
    else:
        Te = Tarr / ev_to_K # [eV]

        # WHITEN THE DATA (Normalize such that the data has unit variance)
        Tew           = scipy.cluster.vq.whiten(Te)
        Tecw0         = Tew[np.random.choice(Tew.shape[0], n_grids, replace=False)]
        # k-means clustering based on temperature
        Tecw          = scipy.cluster.vq.kmeans(Tew, np.linspace(np.min(Tew), np.max(Tew), n_grids), iter=1000, thresh=1e-8, check_finite=False)[0]

        Tecw0[0:len(Tecw)] = Tecw[:]
        Tecw               = Tecw0
        assert len(Tecw0) == n_grids

        Te_b          = np.sort(Tecw * np.std(Te, axis=0))
        dist_mat      = np.zeros((len(Te),n_grids))

        print("rank [%d/%d] : K-means Te clusters "%(rank_, size_), Te_b, flush=True)
        for i in range(n_grids):
            dist_mat[:,i] = np.abs(Tew-Tecw[i])

        membership = np.argmin(dist_mat, axis=1)
        grid_idx_to_spatial_pts_map = list()
        for b_idx in range(n_grids):
            grid_idx_to_spatial_pts_map.append(np.argwhere(membership==b_idx)[:,0])

        grid_idx_to_npts            = np.array([len(a) for a in grid_idx_to_spatial_pts_map], dtype=np.int32)
        grid_idx_to_spatial_idx_map = grid_idx_to_spatial_pts_map

        np.sum(grid_idx_to_npts)    == len(Te), "[Error] : TPS spatial points for v-space grid assignment is inconsitant"

        # grid_idx_to_spatial_pts_map is passed as a vector of int to C++
        # Conver the list to a vector
        grid_pts_to_spatial_index_map_vec = np.zeros(np.sum(grid_idx_to_npts), dtype=grid_idx_to_spatial_idx_map[0].dtype)
        Te = np.array([max(0.1, Te_b[b_idx])  for b_idx in range(n_grids)]) # xp.ones(self.param.n_grids) * self.param.Te 
        glo = 0
        for gidx in range(n_grids):
            ghi = glo + grid_idx_to_npts[gidx]
            grid_pts_to_spatial_index_map_vec[glo:ghi] = grid_idx_to_spatial_idx_map[gidx]
            glo = ghi

    return grid_idx_to_npts, grid_pts_to_spatial_index_map_vec, Te



