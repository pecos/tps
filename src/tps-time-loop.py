#!/usr/bin/env python3
import sys
import os
from mpi4py import MPI
import numpy as np
import scipy.constants
import csv
import matplotlib.pyplot as plt

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
sys.path.append(path + "/../../boltzmann/BESolver/python")
import libtps
from   bte_0d3v_batched import bte_0d3v_batched as BoltzmannSolver
import cupy as cp

class BoltzmannSolverParams():
    sp_order      = 8           # B-spline order in v-space
    spline_qpts   = 10          # number of Gauss-Legendre quadrature points per knot interval    
    Nr            = 127         # number of B-splines used in radial direction
    l_max         = 1           # spherical modes uses, 0, to l_max
    ev_max        = 16          # v-space grid truncation (eV)
    n_grids       = 1           # number of v-space grids

    dt            = 1e-2        # [] non-dimentionalized time w.r.t. oscilation period
    cycles        = 10          # number of max cycles to evolve
    solver_type   = "transient" # two modes, "transient" or "steady-state"
    atol          = 1e-16       # absolute tolerance
    rtol          = 1e-12       # relative tolerance
    max_iter      = 1000         # max iterations for the newton solver

    ee_collisions = 0           # enable electron-electron Coulombic effects
    use_gpu       = 1           # enable GPU use (1)-GPU solver, (0)-CPU solver
    dev_id        = 0           # which GPU device to use only used when use_gpu=1

    collisions    = ["g0","g2"] # collision string g0-elastic, g2-ionization
    export_csv    = 1           # export the qois to csv file
    plot_data     = 1
    
    Efreq         = 0.0 #[1/s]  # E-field osicllation frequency
    verbose       = 1           # verbose output for the BTE solver
    n_pts         = 10          # number of spatial points to launch the BTE solver
    Te            = 0.5 #[eV]   # approximate electron temperature
    
    threads       = 16          # number of threads to use to assemble operators
    grid_idx      = 0
    
    output_dir    = "batched_bte"
    out_fname     = output_dir + "/tps"
    
    # some useful units and conversion factors. 
    ev_to_K       = (scipy.constants.electron_volt/scipy.constants.Boltzmann) 
    Td_fac        = 1e-21 #[Vm^2]
    
class TPSINDEX():
    """
    simple index map to differnt fields, from the TPS arrays
    """
    ION_IDX = 0                         # ion      density index
    ELE_IDX = 1                         # electron density index
    NEU_IDX = 2                         # neutral  density index
    
    EF_RE_IDX = 0                       # Re(E) index
    EF_IM_IDX = 1                       # Im(E) index
    
class Boltzmann0D2VBactchedSolver:
    def __init__(self, tps):
        self.tps   = tps
        self.param = BoltzmannSolverParams()
        # overide the default params, based on the config.ini file.
        self.param.Efreq = 0#tps.getRequiredInput("em/current_frequency")
        self.param.solver_type = "steady-state"
        #self.param.n_pts       = 10
        
        lm_modes        = [[[l,0] for l in range(self.param.l_max+1)]]
        nr              = np.ones(self.param.n_grids, dtype=np.int32) * self.param.Nr
        
        Te              = np.ones(self.param.n_grids) * self.param.Te 
        ev_max          = np.ones(self.param.n_grids) * self.param.ev_max
        self.bte_solver = BoltzmannSolver(self.param, ev_max ,Te , nr, lm_modes, self.param.n_grids, self.param.collisions)
        
        # compute BTE operators
        grid_idx        = self.param.grid_idx
        self.bte_solver.assemble_operators(grid_idx)
        
    def fetch(self, interface):
        grid_idx          = self.param.grid_idx
        Tg                = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=True)
        tps_npts          = len(Tg)
        
        Te                = np.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=True)
        rr                = np.array(interface.HostRead(libtps.t2bIndex.ReactionRates), copy=True)
        efield            = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=True).reshape((2, tps_npts))
        species_densities = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=True).reshape(3, tps_npts)
        
        bte_idx           = Te > (0.4 * self.param.ev_to_K)
        self.param.n_pts  = len(Te[bte_idx])
        
        ni                = species_densities[TPSINDEX.ION_IDX][bte_idx]
        ne                = species_densities[TPSINDEX.ELE_IDX][bte_idx]
        n0                = species_densities[TPSINDEX.NEU_IDX][bte_idx]
        Tg                = Tg[bte_idx]
        Te                = Te[bte_idx]
        
        ne[ne<0]          = 1e-16
        ni[ni<0]          = 1e-16
        
        eRe               = efield[TPSINDEX.EF_RE_IDX][bte_idx]
        eIm               = efield[TPSINDEX.EF_IM_IDX][bte_idx]
        eMag              = np.sqrt(eRe**2 + eIm **2)
        eByn0             = eMag/n0/self.param.Td_fac
        
        
        if self.param.verbose == 1 :
            print("Boltzmann Solver Inputs")
            print("Efreq = %.4E [1/s]" %(self.param.Efreq))
            print("n_pts = %d" % self.param.n_pts)
            # idx0 = np.argmin(eByn0)
            # idx1 = np.argmax(eByn0)
            # print("E/n0  (min)               = %.12E [Td]     \t E/n0 (max) = %.12E [Td]    "%(eByn0[idx0], eByn0[idx1]))
            # print("at E/n0 min max, Tg       = %.12E [K]      \t Tg         = %.12E [K]     "%(Tg[idx0], Tg[idx1]))
            # print("at E/n0 min max, Te       = %.12E [K]      \t Te         = %.12E [K]     "%(Te[idx0], Te[idx1]))
            
            # print("at E/n0 min max, ne       = %.12E [1/m^3]  \t ne         = %.12E [1/m^3] "%(ne[idx0], ne[idx1]))
            # print("at E/n0 min max, ni       = %.12E [1/m^3]  \t ni         = %.12E [1/m^3] "%(ni[idx0], ni[idx1]))
            # print("at E/n0 min max, n0       = %.12E [1/m^3]  \t n0         = %.12E [1/m^3] "%(n0[idx0], n0[idx1]))
            print("E/n0  (min)               = %.12E [Td]         \t E/n0 (max) = %.12E [Td]    "%(np.min(eByn0), np.max(eByn0)))
            print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(np.min(Tg), np.max(Tg)))
            print("Te    (min)               = %.12E [K]          \t Te   (max) = %.12E [K]     "%(np.min(Te), np.max(Te)))
            
            print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(np.min(ne), np.max(ne)))
            print("ni    (min)               = %.12E [1/m^3]      \t ni   (max) = %.12E [1/m^3] "%(np.min(ni), np.max(ni)))
            print("n0    (min)               = %.12E [1/m^3]      \t n0   (max) = %.12E [1/m^3] "%(np.min(n0), np.max(n0)))
            
        
        self.bte_solver.set_boltzmann_parameters(grid_idx, n0, ne, ni, Tg, self.param.solver_type)
        self.bte_f0    = self.bte_solver.initialize(0, self.param.n_pts, "maxwellian")
        if self.param.Efreq == 0:
            ef_t = lambda t : eMag
        else:
            ef_t = lambda t : eRe * np.cos(2 * np.pi * self.param.Efreq * t) + eIm * np.sin(2 * np.pi * self.param.Efreq * t)

        if self.param.use_gpu==1:
            dev_id   = self.param.dev_id
            self.bte_solver.host_to_device_setup(dev_id, 0)
    
            eRe_d     = cp.asarray(eRe)
            eIm_d     = cp.asarray(eIm)
            
            if self.param.Efreq == 0:
                ef_t = lambda t : cp.sqrt(eRe_d**2 + eIm_d**2)
            else:
                ef_t = lambda t : eRe_d * cp.cos(2 * cp.pi * self.param.Efreq * t) + eIm_d * cp.sin(2 * cp.pi * self.param.Efreq * t)
            
            self.bte_f0 = cp.asarray(self.bte_f0)
                
        self.bte_solver.set_efield_function(ef_t)
        return        

    def solve(self):
        grid_idx = self.param.grid_idx
        ff , qoi = self.bte_solver.solve(grid_idx, self.bte_f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
        ev       = np.linspace(1e-3, self.bte_solver._par_ev_range[0][1], 500)
        ff_r     = self.bte_solver.compute_radial_components(grid_idx, ev, ff)

        if self.param.use_gpu==1:
            self.bte_solver.device_to_host_setup(self.param.dev_id,grid_idx)

        ff_r     = cp.asnumpy(ff_r)
        for k, v in qoi.items():
            qoi[k] = cp.asnumpy(v)

        csv_write = self.param.export_csv
        if csv_write:
            fname = self.param.out_fname
            with open("%s_qoi.csv"%fname, 'w', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter=',')
                # write the header
                header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
                for col_idx, g in enumerate(self.param.collisions):
                    header.append(str(g))
                    
                writer.writerow(header)
                
                n0 = self.bte_solver._par_bte_params[grid_idx]["n0"]
                ne = self.bte_solver._par_bte_params[grid_idx]["ne"]
                ni = self.bte_solver._par_bte_params[grid_idx]["ni"]
                Tg = self.bte_solver._par_bte_params[grid_idx]["Tg"]
                
                eRe = self.bte_solver._par_ef_t(0)
                eIm = 0 * self.bte_solver._par_ef_t(0)
                
                if self.param.use_gpu==1:
                    eRe = cp.asnumpy(eRe)
                    eIm = cp.asnumpy(eIm)
                
                eMag  = np.sqrt(eRe**2 + eIm**2)
                data  = np.concatenate((n0.reshape(-1,1), ne.reshape(-1,1), ni.reshape(-1,1), Tg.reshape(-1,1), eMag.reshape(-1,1), qoi["energy"].reshape(-1,1), qoi["mobility"].reshape(-1,1), qoi["diffusion"].reshape(-1,1)), axis=1)
                for col_idx, g in enumerate(self.param.collisions):
                    data = np.concatenate((data, qoi["rates"][col_idx].reshape(-1,1)), axis=1)
                
                writer.writerows(data)


        plot_data    = self.param.plot_data
        if plot_data:
            num_sh       = len(self.bte_solver._par_lm[grid_idx])
            num_subplots = num_sh 
            num_plt_cols = min(num_sh, 4)
            num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
            fig        = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
            plt_idx    =  1
            n_pts_step =  self.param.n_pts // 20

            for lm_idx, lm in enumerate(self.bte_solver._par_lm[grid_idx]):
                plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                for ii in range(0, self.param.n_pts, n_pts_step):
                    fr = np.abs(ff_r[ii, lm_idx, :])
                    plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td], $n_e/n_0$ = %.2E "%(Tg[ii], eMag[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]))
                
                plt.xlabel(r"energy (eV)")
                plt.ylabel(r"$f_%d$"%(lm[0]))
                plt.grid(visible=True)
                if lm_idx==0:
                    plt.legend(prop={'size': 6})
                    
                plt_idx +=1
            
            #plt_idx = num_sh
            plt.savefig("%s_plot.png"%(self.param.out_fname))
        
        
    def push(self, interface):
        pass
        #electron_temperature =  np.array(interface.HostWrite(libtps.t2bIndex.ElectronTemperature), copy=False)
        #electron_temperature[:] = 1.





comm = MPI.COMM_WORLD
# TPS solver
tps = libtps.Tps(comm)

tps.parseCommandLineArgs(sys.argv)
tps.parseInput()
tps.chooseDevices()
tps.chooseSolver()
tps.initialize()

boltzmann = Boltzmann0D2VBactchedSolver(tps)

interface = libtps.Tps2Boltzmann(tps)
tps.initInterface(interface)

it = 0
max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
print("Max Iters: ", max_iters)
tps.solveBegin()
tps.solveStep()
tps.push(interface)
boltzmann.fetch(interface)
boltzmann.solve()

# while it < max_iters:
#     tps.solveStep()
#     tps.push(interface)
#     boltzmann.fetch(interface)
#     boltzmann.solve()
#     boltzmann.push(interface)
#     tps.fetch(interface)
    
#     it = it+1
#     print("it, ", it)

tps.solveEnd()


sys.exit (tps.getStatus())
