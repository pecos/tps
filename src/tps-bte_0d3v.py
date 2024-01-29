#!/usr/bin/env python3
import sys
import os
from mpi4py import MPI
import numpy as np
import scipy.constants
import csv
import matplotlib.pyplot as plt
from time import perf_counter as time
import configparser
import cupy as cp
import enum
import pandas as pd
import scipy.interpolate

class profile_t:
    def __init__(self,name):
        self.name = name
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0

    def __add__(self,o):
        assert(self.name==o.name)
        self.seconds+=o.seconds
        self.snap+=o.snap
        self.iter+=o.iter
        return self

    def start(self):
        self._pri_time = time()
    
    def stop(self):
        self.seconds-=self._pri_time
        self.snap=-self._pri_time

        self._pri_time = time()

        self.seconds +=self._pri_time
        self.snap  += self._pri_time
        self.iter+=1
    
    def reset(self):
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0

def min_mean_max(a, comm: MPI.Comm):
    return (comm.allreduce(a, MPI.MIN) , comm.allreduce(a, MPI.SUM)/comm.Get_size(), comm.allreduce(a, MPI.MAX))


try:
    df    = pd.read_csv("ionization_rates.csv")
    Te    = np.array(df["Te[K]"]) 
    r_arr = np.array(df["Arr[m3/s]"])
    r_csc = np.array(df["CSC_Maxwellian[m3/s]"])
    r_arr = scipy.interpolate.interp1d(Te, r_arr,bounds_error=False, fill_value=0.0)
    r_csc = scipy.interpolate.interp1d(Te, r_csc,bounds_error=False, fill_value=0.0)
    print("ionization coefficient read from file ")
except:
    print("ionization rate coefficient file not found!!")
    r_arr = lambda Te : 1.235e-13 * np.exp(-18.687 / np.abs(Te * scipy.constants.Boltzmann/scipy.constants.electron_volt))
    r_csc = lambda Te : 1.235e-13 * np.exp(-18.687 / np.abs(Te * scipy.constants.Boltzmann/scipy.constants.electron_volt))

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
sys.path.append(path + "/../../boltzmann/BESolver/python")
import libtps
from   bte_0d3v_batched import bte_0d3v_batched as BoltzmannSolver

WITH_PARLA = 1
if WITH_PARLA:
    try:
        from parla import Parla
        from parla.tasks import spawn, TaskSpace
        from parla.devices import cpu, gpu
    except:
        print("Error occured during Parla import. Please make sure Parla is installed properly.")
        sys.exit(0)


class pp(enum.IntEnum):
    SETUP         = 0
    SOLVE         = 1
    LAST          = 2

class BoltzmannSolverParams():
    sp_order      = 3           # B-spline order in v-space
    spline_qpts   = 5           # number of Gauss-Legendre quadrature points per knot interval    
    Nr            = 127         # number of B-splines used in radial direction
    l_max         = 1           # spherical modes uses, 0, to l_max
    ev_max        = 16          # v-space grid truncation (eV)
    n_grids       = 4           # number of v-space grids

    dt            = 1e-3        # [] non-dimentionalized time w.r.t. oscilation period
    cycles        = 10             # number of max cycles to evolve
    solver_type   = "transient" # two modes, "transient" or "steady-state"
    atol          = 1e-10       # absolute tolerance
    rtol          = 1e-10       # relative tolerance
    max_iter      = 1000        # max iterations for the newton solver

    ee_collisions = 0           # enable electron-electron Coulombic effects
    use_gpu       = 1           # enable GPU use (1)-GPU solver, (0)-CPU solver
    dev_id        = 0           # which GPU device to use only used when use_gpu=1

    collisions    = ["g0","g2"] # collision string g0-elastic, g2-ionization
    export_csv    = 1           # export the qois to csv file
    plot_data     = 1
    
    Efreq         = 0.0 #[1/s]  # E-field osicllation frequency
    verbose       = 1           # verbose output for the BTE solver
    Te            = 0.5 #[eV]   # approximate electron temperature
    
    threads       = 16          # number of threads to use to assemble operators
    grid_idx      = 0
    
    output_dir    = "batched_bte1"
    out_fname     = output_dir + "/tps"
    
    # some useful units and conversion factors. 
    ev_to_K       = (scipy.constants.electron_volt/scipy.constants.Boltzmann) 
    Td_fac        = 1e-21 #[Vm^2]
    c_gamma       = np.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass) #[(C/kg)^{1/2}]
    me            = scipy.constants.electron_mass
    kB            = scipy.constants.Boltzmann
    
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
    
    def __init__(self, tps, comm):
        self.tps   = tps
        self.comm : MPI.Comm  = comm
        self.param = BoltzmannSolverParams()
        # overide the default params, based on the config.ini file.
        self.__parse_config_file__(sys.argv[2])
        self.xp_module          = np
        boltzmann_dir           = self.param.output_dir
        isExist = os.path.exists(boltzmann_dir)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(boltzmann_dir)
           #print("directory %s is created!"%(dir_name))
           
        profile_tt  = [None] * int(pp.LAST)
        profile_nn  = ["setup", "solve", "last"]
        for i in range(pp.LAST):
            profile_tt[i] = profile_t(profile_nn[i])
        
        self.profile_tt = profile_tt
        self.profile_nn = profile_nn
        
        num_gpus_per_node = 1 
        if self.param.use_gpu==1:
            num_gpus_per_node = cp.cuda.runtime.getDeviceCount()
        
        # how to map each grid to the GPU devices on the node
        self.gidx_to_device_map = lambda gidx, num_grids : gidx % num_gpus_per_node
        return
    
    def __parse_config_file__(self, fname):
        """
        add the configuaraion file parse code here, 
        which overides the default BoltzmannSolverParams
        """
        config = configparser.ConfigParser()
        print("[Boltzmann] reading configure file given by : ", fname)
        config.read(fname)
        
        self.param.sp_order         = int(config.get("boltzmannSolver", "sp_order").split("#")[0].strip())
        self.param.spline_qpts      = int(config.get("boltzmannSolver", "spline_qpts").split("#")[0].strip())
        
        self.param.Nr               = int(config.get("boltzmannSolver", "Nr").split("#")[0].strip())
        self.param.l_max            = int(config.get("boltzmannSolver", "l_max").split("#")[0].strip())
        self.param.n_grids          = int(config.get("boltzmannSolver", "n_grids").split("#")[0].strip())
        self.param.dt               = float(config.get("boltzmannSolver", "dt").split("#")[0].strip())
        self.param.cycles           = float(config.get("boltzmannSolver", "cycles").split("#")[0].strip())
        self.param.solver_type      = str(config.get("boltzmannSolver", "solver_type").split("#")[0].strip()) 
        self.param.atol             = float(config.get("boltzmannSolver", "atol").split("#")[0].strip())
        self.param.rtol             = float(config.get("boltzmannSolver", "rtol").split("#")[0].strip())
        self.param.max_iter         = int(config.get("boltzmannSolver", "max_iter").split("#")[0].strip())
        self.param.ee_collisions    = int(config.get("boltzmannSolver", "ee_collisions").split("#")[0].strip())
        self.param.use_gpu          = int(config.get("boltzmannSolver", "use_gpu").split("#")[0].strip())
        #self.param.collisions       = config.get("boltzmannSolver", "collisions").split("#")[0]
        
        self.param.export_csv       = int(config.get("boltzmannSolver", "export_csv").split("#")[0].strip())
        self.param.plot_data        = int(config.get("boltzmannSolver", "plot_data").split("#")[0].strip())
        self.param.Efreq            = float(config.get("boltzmannSolver", "Efreq").split("#")[0].strip())
        self.param.verbose          = int(config.get("boltzmannSolver", "verbose").split("#")[0].strip())
        self.param.Te               = float(config.get("boltzmannSolver", "Te").split("#")[0].strip())

        self.param.threads          = int(config.get("boltzmannSolver", "threads").split("#")[0].strip())
        self.param.output_dir       = str(config.get("boltzmannSolver", "output_dir").split("#")[0].strip())
        self.param.out_fname        = self.param.output_dir + "/" + str(config.get("boltzmannSolver", "output_fname").split("#")[0].strip())
        return 
    
    def grid_setup(self, interface):
        """
        Perform the boltzmann grid setup. 
        we generate v-space grid for each spatial point cluster in the parameter space, 
        where, at the moment the clustering is determined based on the electron temperature
        computed from the TPS code. 
        """
        assert self.xp_module==np, "grid setup only supported in CPU"
        self.profile_tt[pp.SETUP].start()
        
        xp                = self.xp_module
        Te                = xp.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False) / self.param.ev_to_K # [eV]
        Te_min, Te_max    = xp.min(Te), xp.max(Te)
        Te_b              = xp.linspace(Te_min, Te_max, self.param.n_grids, endpoint=False)
        dist_mat          = xp.zeros((len(Te), self.param.n_grids))
        
        for iter in range(50):
            #print("clustering iteration ", iter, Te_b)
            for i in range(self.param.n_grids):
                dist_mat[:,i] = xp.abs(Te-Te_b[i])
            
            membership = xp.argmin(dist_mat, axis=1)
            Te_b1      = np.array([np.mean(Te[xp.argwhere(membership==i)[:,0]]) for i in range(self.param.n_grids)])
            rel_error  = np.max(np.abs(1 - Te_b1/Te_b))
            Te_b       = Te_b1
           
            if rel_error < 1e-4:
                break
        Te_b = np.sort(Te_b)
        print("K-means Te clusters ", Te_b)                
        for i in range(self.param.n_grids):
            dist_mat[:,i] = xp.abs(Te-Te_b[i])
        
        membership = xp.argmin(dist_mat, axis=1)
        grid_idx_to_spatial_pts_map = list()
        for b_idx in range(self.param.n_grids):
            #grid_idx_to_spatial_pts_map.append(xp.argwhere(xp.logical_and(Te>= Te_b[b_idx], Te < Te_b[b_idx+1]))[:,0]) 
            grid_idx_to_spatial_pts_map.append(xp.argwhere(membership==b_idx)[:,0]) 
        
        np.save("%s_gidx_to_pidx.npy"%(self.param.out_fname), np.array(grid_idx_to_spatial_pts_map, dtype=object), allow_pickle=True)
        
        self.grid_idx_to_npts            = xp.array([len(a) for a in grid_idx_to_spatial_pts_map], dtype=xp.int32)
        self.grid_idx_to_spatial_idx_map = grid_idx_to_spatial_pts_map
        
        xp.sum(self.grid_idx_to_npts) == len(Te), "[Error] : TPS spatial points for v-space grid assignment is inconsitant"
        lm_modes                         = [[[l,0] for l in range(self.param.l_max+1)] for grid_idx in range(self.param.n_grids)]
        nr                               = xp.ones(self.param.n_grids, dtype=np.int32) * self.param.Nr
        Te                               = xp.array([Te_b[b_idx]  for b_idx in range(self.param.n_grids)]) # xp.ones(self.param.n_grids) * self.param.Te 
        vth                              = np.sqrt(2* self.param.kB * Te * self.param.ev_to_K  /self.param.me)
        ev_max                           = (6 * vth / self.param.c_gamma)**2 
        self.bte_solver                  = BoltzmannSolver(self.param, ev_max ,Te , nr, lm_modes, self.param.n_grids, self.param.collisions)

        if self.param.verbose==1:
            print("grid energy max (eV) \n", ev_max, flush = True)
        
        # compute BTE operators
        for grid_idx in range(self.param.n_grids):
            print("setting up grid %d"%(grid_idx), flush = True)
            self.bte_solver.assemble_operators(grid_idx)
            
        n_grids              = self.param.n_grids
        gidx_to_device_map   = self.gidx_to_device_map

        for grid_idx in range(n_grids):
            assert self.grid_idx_to_npts[grid_idx] > 0

            print("setting initial Maxwellian at %.4E eV" %(self.bte_solver._par_ap_Te[grid_idx]), flush=True)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "f_mw", self.bte_solver.initialize(grid_idx, self.grid_idx_to_npts[grid_idx], "maxwellian"))
            
        
        self.profile_tt[pp.SETUP].stop()
        return

    def solve_wo_parla(self):
        xp                      = self.xp_module
        csv_write               = self.param.export_csv
        plot_data               = self.param.plot_data
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        use_gpu                 = self.param.use_gpu
        dev_id                  = self.param.dev_id
        verbose                 = self.param.verbose
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        self.qoi         = [None for grid_idx in range(self.param.n_grids)]
        self.ff          = [None for grid_idx in range(self.param.n_grids)]
        
        if csv_write: 
            data_csv = np.empty((self.tps_npts, 8 + len(self.param.collisions)))
        
        t1 = time()
        for grid_idx in range(n_grids):
            dev_id   = gidx_to_device_map(grid_idx, n_grids)
            if (use_gpu==0):
                try:
                    f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                    self.qoi[grid_idx] = qoi
                    self.ff [grid_idx] = ff
                except:
                    print("solver failed for v-space gird no %d"%(grid_idx))
                    sys.exit(-1)
            else:
                with xp.cuda.Device(dev_id):
                    try:
                        f0       = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                        ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                        self.qoi[grid_idx] = qoi
                        self.ff [grid_idx] = ff
                    except:
                        print("solver failed for v-space gird no %d"%(grid_idx))
                        sys.exit(-1)
                    
        t2 = time()
        print("time for boltzmann v-space solve = %.4E"%(t2- t1))
        
        if (self.param.export_csv ==1 or self.param.plot_data==1):
            for grid_idx in range(n_grids):
                dev_id   = gidx_to_device_map(grid_idx, n_grids)
                ff       = self.ff[grid_idx]
                qoi      = self.qoi[grid_idx]
                
                def asnumpy(a):
                    if cp.get_array_module(a)==cp:
                        with cp.cuda.Device(dev_id):
                            return cp.asnumpy(a)
                    else:
                        return a
                
                ff_cpu   = asnumpy(ff)
                ev       = np.linspace(1e-3, self.bte_solver._par_ev_range[grid_idx][1], 500)
                ff_r     = self.bte_solver.compute_radial_components(grid_idx, ev, ff_cpu)
                
                n0    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "n0"))
                ne    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ne"))
                ni    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ni"))
                Tg    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "Tg"))
                eRe   = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe"))
                eIm   = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm"))
                eMag  = np.sqrt(eRe**2 + eIm**2)
                
                if csv_write:
                    data_csv[gidx_to_pidx_map[grid_idx], 0]    = n0
                    data_csv[gidx_to_pidx_map[grid_idx], 1]    = ne
                    data_csv[gidx_to_pidx_map[grid_idx], 2]    = ni
                    data_csv[gidx_to_pidx_map[grid_idx], 3]    = Tg
                    data_csv[gidx_to_pidx_map[grid_idx], 4]    = eMag
                    data_csv[gidx_to_pidx_map[grid_idx], 5]    = asnumpy(qoi["energy"])
                    data_csv[gidx_to_pidx_map[grid_idx], 6]    = asnumpy(qoi["mobility"])
                    data_csv[gidx_to_pidx_map[grid_idx], 7]    = asnumpy(qoi["diffusion"])
                    
                    for col_idx, g in enumerate(self.param.collisions):
                        data_csv[gidx_to_pidx_map[grid_idx], 8 + col_idx]    = asnumpy(qoi["rates"][col_idx])

                if plot_data:
                    num_sh       = len(self.bte_solver._par_lm[grid_idx])
                    num_subplots = num_sh 
                    num_plt_cols = min(num_sh, 4)
                    num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
                    fig          = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
                    plt_idx      =  1
                    n_pts_step   =  self.grid_idx_to_npts[grid_idx] // 20

                    for lm_idx, lm in enumerate(self.bte_solver._par_lm[grid_idx]):
                        plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                        for ii in range(0, self.grid_idx_to_npts[grid_idx], n_pts_step):
                            fr = np.abs(ff_r[ii, lm_idx, :])
                            plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td], $n_e/n_0$ = %.2E "%(Tg[ii], eMag[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]))
                        
                        plt.xlabel(r"energy (eV)")
                        plt.ylabel(r"$f_%d$"%(lm[0]))
                        plt.grid(visible=True)
                        if lm_idx==0:
                            plt.legend(prop={'size': 6})
                            
                        plt_idx +=1
                    
                    plt.savefig("%s_plot_%02d.png"%(self.param.out_fname, grid_idx))
                    plt.close()
            
            if csv_write:
                fname    = self.param.out_fname
                with open("%s_qoi.csv"%fname, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f,delimiter=',')
                    # write the header
                    header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
                    for col_idx, g in enumerate(self.param.collisions):
                        header.append(str(g))
                    
                    writer.writerow(header)
                    writer.writerows(data_csv)

        return
    
    async def fetch(self, interface):
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        heavy_temp              = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts                = len(heavy_temp)
        self.tps_npts           = tps_npts
        electron_temp           = np.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False)
        efield                  = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False).reshape((2, tps_npts))
        species_densities       = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(3, tps_npts)
        
        n_grids                 = self.param.n_grids 
        use_gpu                 = self.param.use_gpu
        n_grids                 = self.param.n_grids
        
        gidx_to_device_map      = self.gidx_to_device_map
        
        for grid_idx in range(n_grids):
            bte_idx           = gidx_to_pidx_map[grid_idx]
            dev_id            = self.gidx_to_device_map(grid_idx, n_grids)
            
            ni                = species_densities[TPSINDEX.ION_IDX][bte_idx]
            ne                = species_densities[TPSINDEX.ELE_IDX][bte_idx]
            n0                = species_densities[TPSINDEX.NEU_IDX][bte_idx]
            Tg                = heavy_temp[bte_idx]
            Te                = electron_temp[bte_idx]
            eRe               = efield[TPSINDEX.EF_RE_IDX][bte_idx]
            eIm               = efield[TPSINDEX.EF_IM_IDX][bte_idx]
            
            eMag              = np.sqrt(eRe**2 + eIm **2)
            eByn0             = eMag/n0/self.param.Td_fac
            
            # if self.param.verbose == 1 :
            #     print("Boltzmann solver inputs for v-space grid id %d"%(grid_idx))
            #     print("Efreq = %.4E [1/s]" %(self.param.Efreq))
            #     print("n_pts = %d" % self.grid_idx_to_npts[grid_idx])
                
            #     print("E/n0  (min)               = %.12E [Td]         \t E/n0 (max) = %.12E [Td]    "%(np.min(eByn0), np.max(eByn0)))
            #     print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(np.min(Tg)   , np.max(Tg)))
            #     print("Te    (min)               = %.12E [K]          \t Te   (max) = %.12E [K]     "%(np.min(Te)   , np.max(Te)))
                
            #     print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(np.min(ne)   , np.max(ne)))
            #     print("ni    (min)               = %.12E [1/m^3]      \t ni   (max) = %.12E [1/m^3] "%(np.min(ni)   , np.max(ni)))
            #     print("n0    (min)               = %.12E [1/m^3]      \t n0   (max) = %.12E [1/m^3] "%(np.min(n0)   , np.max(n0)))
            
            
            if (use_gpu == 1):
                with cp.cuda.Device(dev_id):
                    ne        = cp.array(ne)
                    ni        = cp.array(ni)
                    n0        = cp.array(n0)
                    Tg        = cp.array(Tg)
                    Te        = cp.array(Te)
                    eRe       = cp.array(eRe)
                    eIm       = cp.array(eIm)
                    
                    eMag      = cp.sqrt(eRe**2 + eIm **2)
                    eByn0     = eMag/n0/self.param.Td_fac
            
            self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", eRe)
            self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", eIm)
            self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , eMag)
        
            
        return        

    async def solve_init(self):
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        ts = TaskSpace("T")
        for grid_idx in range(self.param.n_grids):
            @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
            def t1():
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                print("[%d/%d] setting grid %d to device %d"%(rank, npes, grid_idx, dev_id))
                self.bte_solver.host_to_device_setup(dev_id, grid_idx)
            
        await ts
        
        def ts_op_setup(grid_idx):
            xp                                      = self.xp_module 
            f_mw                                    = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
            n_pts                                   = f_mw.shape[1]
            Qmat                                    = self.bte_solver._op_qmat[grid_idx]
            INr                                     = xp.eye(Qmat.shape[1])
            self.bte_solver._op_imat_vx[grid_idx]   = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr)
            
        if(self.param.use_gpu==1):
            self.xp_module = cp
            ts = TaskSpace("T")
            
            for grid_idx in range(self.param.n_grids):
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                @spawn(ts[grid_idx], placement=[gpu(dev_id)], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    f_mw = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", f_mw)
                    
            
            await ts
        else:
            self.xp_module = np
            ts = TaskSpace("T")
            for grid_idx in range(self.param.n_grids):
                @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    f_mw = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", f_mw)
            
            await ts
        
        return
            
    async def solve_step(self, time, delta_t):
        """
        perform a single timestep in 0d-BTE
        """
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
        # xp                      = self.xp_module
        # csv_write               = self.param.export_csv
        # plot_data               = self.param.plot_data
        # gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        # use_gpu                 = self.param.use_gpu
        # dev_id                  = self.param.dev_id
        # verbose                 = self.param.verbose
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        ts = TaskSpace("T")
        for grid_idx in range(n_grids):
            @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
            def t1():
                u0    = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                v     = self.bte_solver.step(grid_idx, u0, self.param.atol, self.param.rtol, self.param.max_iter, time, delta_t)
                self.bte_solver.set_boltzmann_parameter(grid_idx, "u1", v)
        await ts
        
        return 
    
    async def solve(self):
        """
        Can be used to compute steady-state or cycle averaged BTE solutions
        """
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
        xp                      = self.xp_module
        csv_write               = self.param.export_csv
        plot_data               = self.param.plot_data
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        use_gpu                 = self.param.use_gpu
        dev_id                  = self.param.dev_id
        verbose                 = self.param.verbose
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        self.qoi                = [None for grid_idx in range(self.param.n_grids)]
        self.ff                 = [None for grid_idx in range(self.param.n_grids)]
        num_gpus                = len(gpu)
        
        if (use_gpu==1):
            parla_placement = [gpu(gidx_to_device_map(grid_idx,n_grids)) for grid_idx in range(n_grids)]
        else:
            parla_placement = [cpu for grid_idx in range(n_grids)]

        if csv_write: 
            data_csv = np.empty((self.tps_npts, 8 + len(self.param.collisions)))
            
        self.profile_tt[pp.SOLVE].start()
        ts = TaskSpace("T")
        for grid_idx in range(self.param.n_grids):
            @spawn(ts[grid_idx], placement=[parla_placement[grid_idx]], vcus=0.0)
            def t1():
                try:
                    print("[Boltzmann] %d / %d launching grid %d on %s"%(rank, npes, grid_idx, parla_placement[grid_idx]))
                    f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                    self.ff[grid_idx]  = ff
                    self.qoi[grid_idx] = qoi
                except:
                    print("solver failed for v-space gird no %d"%(grid_idx))
                    sys.exit(-1)
                    
        await ts
        self.profile_tt[pp.SOLVE].stop()
        
        
        t1 = min_mean_max(self.profile_tt[pp.SETUP].seconds, self.comm)
        t2 = min_mean_max(self.profile_tt[pp.SOLVE].seconds, self.comm)
        print("[Boltzmann] setup (min) = %.4E (s) setup (mean) = %.4E (s) setup (max) = %.4E (s)" % (t1[0],t1[1],t1[2]))
        print("[Boltzmann] solve (min) = %.4E (s) solve (mean) = %.4E (s) solve (max) = %.4E (s)" % (t2[0],t2[1],t2[2]))
        
        if (self.param.export_csv ==1 or self.param.plot_data==1):
            for grid_idx in range(n_grids):
                dev_id   = gidx_to_device_map(grid_idx, n_grids)
                ff       = self.ff[grid_idx]
                qoi      = self.qoi[grid_idx]
                
                def asnumpy(a):
                    if cp.get_array_module(a)==cp:
                        with cp.cuda.Device(dev_id):
                            return cp.asnumpy(a)
                    else:
                        return a
                
                ff_cpu   = asnumpy(ff)
                ev       = np.linspace(1e-3, self.bte_solver._par_ev_range[grid_idx][1], 500)
                ff_r     = self.bte_solver.compute_radial_components(grid_idx, ev, ff_cpu)
                
                n0    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "n0"))
                ne    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ne"))
                ni    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ni"))
                Tg    = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "Tg"))
                eRe   = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe"))
                eIm   = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm"))
                eMag  = np.sqrt(eRe**2 + eIm**2)
                
                if csv_write:
                    data_csv[gidx_to_pidx_map[grid_idx], 0]    = n0
                    data_csv[gidx_to_pidx_map[grid_idx], 1]    = ne
                    data_csv[gidx_to_pidx_map[grid_idx], 2]    = ni
                    data_csv[gidx_to_pidx_map[grid_idx], 3]    = Tg
                    data_csv[gidx_to_pidx_map[grid_idx], 4]    = eMag
                    data_csv[gidx_to_pidx_map[grid_idx], 5]    = asnumpy(qoi["energy"])
                    data_csv[gidx_to_pidx_map[grid_idx], 6]    = asnumpy(qoi["mobility"])
                    data_csv[gidx_to_pidx_map[grid_idx], 7]    = asnumpy(qoi["diffusion"])
                    
                    for col_idx, g in enumerate(self.param.collisions):
                        data_csv[gidx_to_pidx_map[grid_idx], 8 + col_idx]    = asnumpy(qoi["rates"][col_idx])

                if plot_data:
                    num_sh       = len(self.bte_solver._par_lm[grid_idx])
                    num_subplots = num_sh 
                    num_plt_cols = min(num_sh, 4)
                    num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
                    fig          = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
                    plt_idx      =  1
                    n_pts_step   =  self.grid_idx_to_npts[grid_idx] // 20

                    for lm_idx, lm in enumerate(self.bte_solver._par_lm[grid_idx]):
                        plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                        for ii in range(0, self.grid_idx_to_npts[grid_idx], n_pts_step):
                            fr = np.abs(ff_r[ii, lm_idx, :])
                            plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td], $n_e/n_0$ = %.2E "%(Tg[ii], eMag[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]))
                        
                        plt.xlabel(r"energy (eV)")
                        plt.ylabel(r"$f_%d$"%(lm[0]))
                        plt.grid(visible=True)
                        if lm_idx==0:
                            plt.legend(prop={'size': 6})
                            
                        plt_idx +=1
                    
                    plt.savefig("%s_plot_%02d.png"%(self.param.out_fname, grid_idx))
                    plt.close()
            
            if csv_write:
                fname    = self.param.out_fname
                with open("%s_qoi.csv"%fname, 'w', encoding='UTF8') as f:
                    writer = csv.writer(f,delimiter=',')
                    # write the header
                    header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
                    for col_idx, g in enumerate(self.param.collisions):
                        header.append(str(g))
                    
                    writer.writerow(header)
                    writer.writerows(data_csv)

        return
    
    async def push(self, interface):
        xp                      = self.xp_module
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        
        heavy_temp  = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts    = len(heavy_temp)
        
        n_reactions = interface.nComponents(libtps.t2bIndex.ReactionRates)
        rates       = np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False).reshape((n_reactions, tps_npts))
        
        if(n_reactions>0):
            ts = TaskSpace("T")
            for grid_idx in range(n_grids):
                @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                def t1():
                    qA       = boltzmann.bte_solver._op_diag_dg[grid_idx]
                    u0       = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                    
                    h_curr   = xp.dot(qA, u0)
                    h_curr   = boltzmann.bte_solver.normalized_distribution(grid_idx, h_curr)
                    qoi      = boltzmann.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                    
                    rates[0][gidx_to_pidx_map[grid_idx]] = xp.asnumpy(qoi["rates"][1])
                    
                    
            await ts
        return 
        
if __name__=="__main__":
    comm = MPI.COMM_WORLD
    
    with Parla():
        # TPS solver
        tps = libtps.Tps(comm)
        tps.parseCommandLineArgs(sys.argv)
        tps.parseInput()
        tps.chooseDevices()
        tps.chooseSolver()
        tps.initialize()

        boltzmann = Boltzmann0D2VBactchedSolver(tps, comm)
        interface = libtps.Tps2Boltzmann(tps)
        tps.initInterface(interface)

        #coords = np.array(interface.HostReadSpatialCoordinates(), copy=False)
        tps.solveBegin()
        tps.push(interface)
        boltzmann.grid_setup(interface)
        
        @spawn(placement=cpu, vcus=0)
        async def __main__():
            await boltzmann.solve_init()    
            xp = boltzmann.bte_solver.xp_module

            max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
            iter      = 0
            tt        = 0
            tau       = (1/boltzmann.param.Efreq)
            dt        = 5e-3 * tau
            
            while (iter<max_iters):
                t1 = time()
                tps.solveStep()
                t2 = time()
                
                t1 = min_mean_max(t2-t1, comm)
                print("[TPS] simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (tt/tau, t1[0],t1[1],t1[2]))
                
                tps.push(interface)
                await boltzmann.fetch(interface)
                
                t1 = time()
                await boltzmann.solve_step(tt, dt)
                for grid_idx in range(boltzmann.param.n_grids):
                    u1 = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1")
                    boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u0", u1)
                t2 = time()
                t1 = min_mean_max(t2-t1, comm)
                print("[BTE] simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (tt/tau, t1[0],t1[1],t1[2]))
                
                await boltzmann.push(interface)
                tps.fetch(interface)
                tt += dt
        

    tps.solveEnd()
    sys.exit (tps.getStatus())