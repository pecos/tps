#!/usr/bin/env python3
from mpi4py import MPI
import sys
import os
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
import scipy.cluster
import threading
import datetime
import h5py
# Use asynchronous stream ordered memory
#cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

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

# set path to C++ TPS library
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path + "/.libs")
sys.path.append(path + "/../../boltzmann/BESolver/python")
import libtps
from   bte_0d3v_batched import bte_0d3v_batched as BoltzmannSolver
import utils as bte_utils

WITH_PARLA = 0
if WITH_PARLA:
    try:
        from parla import Parla
        from parla.tasks import spawn, TaskSpace
        from parla.devices import cpu, gpu
    except Exception as e:
        print(e)
        print("Error occured during Parla import. Please make sure Parla is installed properly.")
        sys.exit(0)


class BoltzmannSolverParams():
    sp_order      = 3           # B-spline order in v-space
    spline_qpts   = 5           # number of Gauss-Legendre quadrature points per knot interval    
    Nr            = 127         # number of B-splines used in radial direction
    l_max         = 1           # spherical modes uses, 0, to l_max
    ev_max        = 16          # v-space grid truncation (eV)
    n_grids       = 4           # number of v-space grids
    n_sub_clusters= 200         # number of sub-clusters

    dt            = 1e-3        # [] non-dimentionalized time w.r.t. oscilation period
    cycles        = 10             # number of max cycles to evolve
    solver_type   = "transient" # two modes, "transient" or "steady-state"
    atol          = 1e-10       # absolute tolerance
    rtol          = 1e-10       # relative tolerance
    max_iter      = 1000        # max iterations for the newton solver
    
    tps_bte_max_iter = 5000     # max iterations for tps-bte split scheme
    bte_solve_freq   = 100      # run bte every x tps cycles. 

    ee_collisions = 0           # enable electron-electron Coulombic effects
    use_gpu       = 1           # enable GPU use (1)-GPU solver, (0)-CPU solver
    dev_id        = 0           # which GPU device to use only used when use_gpu=1

    collisions    = ""          # collision string g0-elastic, g2-ionization
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
    N_Avo         = scipy.constants.Avogadro
    
    n0            = 3.22e22 #[m^{-3}]
    
    rand_seed       = 0
    use_clstr_inp   = True
    clstr_maxiter   = 10
    clstr_threshold = 1e-3
    
    EMag_threshold  = 1e-10
    
class TPSINDEX():
    """
    simple index map to differnt fields, from the TPS arrays
    """
    EF_RE_IDX = 0                       # Re(E) index
    EF_IM_IDX = 1                       # Im(E) index
    
    # in future we need to setup this methodically
    # here key denotes the idx running from 0, nreactions-1
    # value denotes the reaction index in the qoi array
    RR_IDX    = {0 : 2 , 1 : 4 , 2 : 1, 3 : 3}
    
    ION_IDX  = 1
    ELE_IDX  = 2
    NEU_IDX  = 3
    EX1_IDX  = 0
    
    MOLE_FRAC_IDX = {0: NEU_IDX, 1: EX1_IDX} 

def k_means(x, num_clusters, xi=None, max_iter=1000, thresh=1e-12, rand_seed=0, xp=np):
    assert x.ndim == 2, "observations must me 2d array"
    if xi is None:
        xp.random.seed(rand_seed)
        xi = x[xp.random.choice(x.shape[0], num_clusters, replace=False)]

    distortion_0 = xp.zeros(num_clusters)
    idx          = xp.arange(num_clusters)
    for iter in range(max_iter):
        distance     = xp.linalg.norm(x[:, None, :] - xi[None, :, :], axis=2)
        pred         = xp.argmin(distance, axis=1)
        mask         = pred == idx[:, None]
        # print(mask)
        # print(xp.where(mask[:, :, None], x, 0))
        # print(xp.where(mask[:, :, None], x, 0).sum(axis=1))
        # print(xp.where(mask[:, :, None], x, 0).shape)
        sums         = xp.where(mask[:, :, None], x, 0).sum(axis=1)
        counts       = xp.count_nonzero(mask, axis=1)
        counts[counts==0] = 1
        # print(distance)
        # print(mask)
        # print(xp.where(mask.T, distance, 0))
        # print(xp.where(mask.T, distance, 0).sum(axis=0).shape)
        # print(counts)
        distortion_1 = xp.where(mask.T, distance, 0).sum(axis=0)/counts[:,None]
        rel_error    = xp.linalg.norm(distortion_0-distortion_1)/xp.linalg.norm(distortion_1)
        #print(iter, rel_error, distortion_1, xi)
        #print(iter, rel_error, xi[0:3])
        if rel_error < thresh:
            break
        
        xi_new       = sums / counts[:,None]
        distortion_0 = distortion_1
        # rel_error = xp.max(xp.linalg.norm(xi_new-xi, axis=1)/xp.linalg.norm(xi, axis=1))
        # if rel_error < thresh:
        #     # print(iter, rel_error)
        #     # print("xi_new", xi_new[0:10,:], "xi", xi[0:10,:])
        #     break
        
        xi = xi_new
        
        
    return xi, pred
    
    
class Boltzmann0D2VBactchedSolver:
    
    def __init__(self, tps, comm):
        self.tps   = tps
        self.comm : MPI.Comm  = comm
        
        self.rankG = self.comm.Get_rank()
        self.npesG = self.comm.Get_size()
        
        self.param = BoltzmannSolverParams()
        # overide the default params, based on the config.ini file.
        self.__parse_config_file__(sys.argv[2])
        self.xp_module          = np
        boltzmann_dir           = self.param.output_dir
        
        if (self.rankG==0):
            isExist = os.path.exists(boltzmann_dir)
            if not isExist:
                os.makedirs(boltzmann_dir)
           
        num_gpus_per_node = 1 
        if self.param.use_gpu==1:
            num_gpus_per_node = cp.cuda.runtime.getDeviceCount()
        
        self.num_gpus_per_node = num_gpus_per_node 
        
        if self.rankG==0:
            print("number of GPUs detected = %d "%(num_gpus_per_node), flush=True)
        
        # how to map each grid to the GPU devices on the node
        self.gidx_to_device_map = lambda gidx, num_grids : self.rankG % self.num_gpus_per_node #gidx % num_gpus_per_node
        return
    
    def __parse_config_file__(self, fname):
        """
        add the configuaraion file parse code here, 
        which overides the default BoltzmannSolverParams
        """
        config = configparser.ConfigParser()
        if self.rankG==0:
            print("[Boltzmann] reading configure file given by : ", fname, flush=True)
            
        config.read(fname)
        
        self.param.sp_order         = int(config.get("boltzmannSolver", "sp_order").split("#")[0].strip())
        self.param.spline_qpts      = int(config.get("boltzmannSolver", "spline_qpts").split("#")[0].strip())
        
        self.param.Nr               = int(config.get("boltzmannSolver", "Nr").split("#")[0].strip())
        self.param.l_max            = int(config.get("boltzmannSolver", "l_max").split("#")[0].strip())
        self.param.n_grids          = int(config.get("boltzmannSolver", "n_grids").split("#")[0].strip())
        self.param.n_sub_clusters   = int(config.get("boltzmannSolver", "n_sub_clusters").split("#")[0].strip())
        self.param.dt               = float(config.get("boltzmannSolver", "dt").split("#")[0].strip())
        self.param.cycles           = float(config.get("boltzmannSolver", "cycles").split("#")[0].strip())
        self.param.solver_type      = str(config.get("boltzmannSolver", "solver_type").split("#")[0].strip()) 
        self.param.atol             = float(config.get("boltzmannSolver", "atol").split("#")[0].strip())
        self.param.rtol             = float(config.get("boltzmannSolver", "rtol").split("#")[0].strip())
        self.param.max_iter         = int(config.get("boltzmannSolver", "max_iter").split("#")[0].strip())
        self.param.ee_collisions    = int(config.get("boltzmannSolver", "ee_collisions").split("#")[0].strip())
        self.param.use_gpu          = int(config.get("boltzmannSolver", "use_gpu").split("#")[0].strip())
        self.param.collisions       = str(config.get("boltzmannSolver", "collisions").split("#")[0].strip())
        
        self.param.export_csv       = int(config.get("boltzmannSolver", "export_csv").split("#")[0].strip())
        self.param.plot_data        = int(config.get("boltzmannSolver", "plot_data").split("#")[0].strip())
        self.param.Efreq            = float(config.get("boltzmannSolver", "Efreq").split("#")[0].strip())
        self.param.verbose          = int(config.get("boltzmannSolver", "verbose").split("#")[0].strip())
        self.param.Te               = float(config.get("boltzmannSolver", "Te").split("#")[0].strip())

        self.param.threads          = int(config.get("boltzmannSolver", "threads").split("#")[0].strip())
        self.param.output_dir       = str(config.get("boltzmannSolver", "output_dir").split("#")[0].strip())
        self.param.out_fname        = self.param.output_dir + "/" + str(config.get("boltzmannSolver", "output_fname").split("#")[0].strip())
        
        self.param.bte_solve_freq   = int(config.get("boltzmannSolver", "bte_solve_freq").split("#")[0].strip())
        self.param.tps_bte_max_iter = int(config.get("boltzmannSolver", "tps_bte_max_iter").split("#")[0].strip())
        
        return 
    
    def grid_setup(self, interface):
        """
        Perform the boltzmann grid setup. 
        we generate v-space grid for each spatial point cluster in the parameter space, 
        where, at the moment the clustering is determined based on the electron temperature
        computed from the TPS code. 
        """
        assert self.xp_module==np, "grid setup only supported in CPU"
        xp            = self.xp_module
        n_grids       = self.param.n_grids
        Te            = xp.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False) / self.param.ev_to_K # [eV]
        
        Tew           = scipy.cluster.vq.whiten(Te)
        Tecw0         = Tew[np.random.choice(Tew.shape[0], self.param.n_grids, replace=False)]
        Tecw          = scipy.cluster.vq.kmeans(Tew, np.linspace(np.min(Tew), np.max(Tew), n_grids), iter=1000, thresh=1e-8, check_finite=False)[0]
        
        Tecw0[0:len(Tecw)] = Tecw[:]
        Tecw               = Tecw0
        assert len(Tecw0) == self.param.n_grids
        
        Te_b          = xp.sort(Tecw * np.std(Te, axis=0))
        dist_mat      = xp.zeros((len(Te),n_grids))
        
        
        print("rank [%d/%d] : K-means Te clusters "%(self.rankG, self.npesG), Te_b, flush=True)
        for i in range(self.param.n_grids):
            dist_mat[:,i] = xp.abs(Tew-Tecw[i])
        
        membership = xp.argmin(dist_mat, axis=1)
        grid_idx_to_spatial_pts_map = list()
        for b_idx in range(self.param.n_grids):
            grid_idx_to_spatial_pts_map.append(xp.argwhere(membership==b_idx)[:,0]) 
        
        #np.save("%s_gidx_to_pidx_rank_%08d.npy"%(self.param.out_fname, self.rankG), np.array(grid_idx_to_spatial_pts_map, dtype=object), allow_pickle=True)
        
        self.grid_idx_to_npts            = xp.array([len(a) for a in grid_idx_to_spatial_pts_map], dtype=xp.int32)
        self.grid_idx_to_spatial_idx_map = grid_idx_to_spatial_pts_map
        
        xp.sum(self.grid_idx_to_npts)    == len(Te), "[Error] : TPS spatial points for v-space grid assignment is inconsitant"
        lm_modes                         = [[[l,0] for l in range(self.param.l_max+1)] for grid_idx in range(self.param.n_grids)]
        nr                               = xp.ones(self.param.n_grids, dtype=np.int32) * self.param.Nr
        Te                               = xp.array([Te_b[b_idx]  for b_idx in range(self.param.n_grids)]) # xp.ones(self.param.n_grids) * self.param.Te 
        vth                              = np.sqrt(2* self.param.kB * Te * self.param.ev_to_K  /self.param.me)
        ev_max                           = (6 * vth / self.param.c_gamma)**2 
        self.bte_solver                  = BoltzmannSolver(self.param, ev_max , Te , nr, lm_modes, self.param.n_grids, self.param.collisions)

        # compute BTE operators
        for grid_idx in range(self.param.n_grids):
            self.bte_solver.assemble_operators(grid_idx)
            
        n_grids              = self.param.n_grids
        gidx_to_device_map   = self.gidx_to_device_map

        for grid_idx in range(n_grids):
            assert self.grid_idx_to_npts[grid_idx] > 0
            if(self.param.use_clstr_inp==True):
                self.bte_solver.set_boltzmann_parameter(grid_idx, "f_mw", self.bte_solver.initialize(grid_idx, self.param.n_sub_clusters      , "maxwellian"))
            else:
                self.bte_solver.set_boltzmann_parameter(grid_idx, "f_mw", self.bte_solver.initialize(grid_idx, self.grid_idx_to_npts[grid_idx], "maxwellian"))
        
        self.comm.Barrier()
        # active_grid_idx=list()
        # for grid_idx in range(n_grids):
        #     spec_sp     = self.bte_solver._op_spec_sp[grid_idx]
        #     ev_max_ext  = (spec_sp._basis_p._t[-1] * vth[grid_idx] / self.param.c_gamma)**2
        #     if  ev_max_ext > 15.76:
        #         active_grid_idx.append(grid_idx)
        
        # self.active_grid_idx  = active_grid_idx #[i for i in range(self.param.n_grids)]
        self.active_grid_idx  = [i for i in range(self.param.n_grids)]
        self.sub_clusters_run = False
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
        coll_list        = self.bte_solver.get_collision_list()
        coll_names       = self.bte_solver.get_collision_names()
        
        if csv_write: 
            data_csv = np.empty((self.tps_npts, 8 + len(coll_list)))
        
        t1 = time()
        for grid_idx in range(n_grids):
            dev_id   = gidx_to_device_map(grid_idx, n_grids)
            if (use_gpu==0):
                try:
                    f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                    self.qoi[grid_idx] = qoi
                    self.ff [grid_idx] = ff
                except Exception as e:
                    print(e)
                    print("solver failed for v-space gird no %d"%(grid_idx), flush=True)
                    self.comm.Abort(0)
            else:
                with xp.cuda.Device(dev_id):
                    try:
                        f0       = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                        ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                        self.qoi[grid_idx] = qoi
                        self.ff [grid_idx] = ff
                    except Exception as e:
                        print(e)
                        print("solver failed for v-space gird no %d"%(grid_idx), flush=True)
                        self.comm.Abort(0)
                    
        t2 = time()
        print("time for boltzmann v-space solve = %.4E"%(t2- t1), flush=True)
        
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
                    
                    for col_idx, g in enumerate(coll_list):
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
                    for col_idx, g in enumerate(coll_list):
                        header.append(str(coll_names[col_idx]))
                    
                    writer.writerow(header)
                    writer.writerows(data_csv)

        return
    
    def fetch(self, interface):
        xp                      = self.xp_module
        use_interp              = self.param.use_clstr_inp
        gidx_to_pidx            = self.grid_idx_to_spatial_idx_map
        heavy_temp              = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts                = len(heavy_temp)
        self.tps_npts           = tps_npts
        nspecies                = interface.Nspecies()
        electron_temp           = np.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False)
        efield                  = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False).reshape((2, tps_npts))
        species_densities       = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(nspecies, tps_npts)
        
        cs_avail_species        = self.bte_solver._avail_species
        
        n0                      = np.sum(species_densities, axis=0) - species_densities[TPSINDEX.ELE_IDX]
        ns_by_n0                = np.concatenate([species_densities[TPSINDEX.MOLE_FRAC_IDX[i]]/n0 for i in range(len(cs_avail_species))]).reshape((len(cs_avail_species), tps_npts))
        
        n_grids                 = self.param.n_grids 
        use_gpu                 = self.param.use_gpu
        
        Tg                      = heavy_temp
        
        Ex                      = efield[0]
        Ey                      = efield[1]
        
        ne                      = species_densities[TPSINDEX.ELE_IDX]
        coll_list               = self.bte_solver.get_collision_list()
        coll_names              = self.bte_solver.get_collision_names()
        cs_data                 = self.bte_solver.get_cross_section_data()
        
        cs_species              = list()
        for col_idx, (k,v) in enumerate(cs_data.items()):
            cs_species.append(v["species"])
        
        cs_species = list(sorted(set(cs_species), key=cs_species.index))
        data_csv   = np.concatenate([(Ex).reshape((-1, 1)),
                                     (Ey).reshape((-1, 1)),
                                     (Tg).reshape((-1, 1)),
                                     (ne/n0).reshape((-1, 1))] + [ns_by_n0[i].reshape((-1, 1)) for i in range(ns_by_n0.shape[0])] + [n0.reshape(-1, 1)], axis=1)
        
        for grid_idx in self.active_grid_idx:
            with open("%s/%s.csv"%(self.param.output_dir, "tps_fetch_grid_%02d_rank_%02d_npes_%02d"%(grid_idx, self.rankG, self.npesG)), 'w', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter=',')
                # write the header
                header = ["eRe", "eIm", "Tg", "ne/n0"] + ["(%s)/n0"%(s) for s in cs_species] + ["n0"]
                writer.writerow(header)
                writer.writerows(data_csv[gidx_to_pidx[grid_idx]])
                
        EMag                    = np.sqrt(Ex**2 + Ey**2)
        e_idx                   = EMag<self.param.EMag_threshold
        
        ExbyN                   = Ex/n0/self.param.Td_fac
        EybyN                   = Ey/n0/self.param.Td_fac
        
        ExbyN[e_idx]            = (self.param.EMag_threshold/np.sqrt(2)) / n0[e_idx] / self.param.Td_fac
        EybyN[e_idx]            = (self.param.EMag_threshold/np.sqrt(2)) / n0[e_idx] / self.param.Td_fac
        
        Ex                      = ExbyN * self.param.n0 * self.param.Td_fac
        Ey                      = EybyN * self.param.n0 * self.param.Td_fac
        
        ion_deg                 = species_densities[TPSINDEX.ELE_IDX]/n0
        ion_deg[ion_deg<=0]     = 1e-16
        ns_by_n0[ns_by_n0<=0]   = 0
        m_bte                   = np.concatenate([ExbyN.reshape((-1, 1)), EybyN.reshape((-1, 1)), Tg.reshape((-1, 1)), ion_deg.reshape((-1, 1))] + [ ns_by_n0[i].reshape((-1, 1)) for i in range(ns_by_n0.shape[0])], axis=1)
        
        self.m_bte              = m_bte
        self.sub_cluster_c      = None
        self.sub_cluster_c_lbl  = None
        gidx_to_device_map      = self.gidx_to_device_map
        
        if (use_interp == True):
            n_sub_clusters               = self.param.n_sub_clusters
            self.sub_cluster_c           = [None for i in range(self.param.n_grids)]
            self.sub_cluster_c_lbl       = [None for i in range(self.param.n_grids)]
            
            def normalize(obs, xp):
                std_obs   = xp.std(obs, axis=0)
                std_obs[std_obs == 0.0] = 1.0
                return obs/std_obs, std_obs

            for grid_idx in self.active_grid_idx:
                dev_id    = self.gidx_to_device_map(grid_idx, n_grids)
                
                def t1():
                    # xp                           = cp
                    # m                            = xp.array(m_bte[gidx_to_pidx[grid_idx]])
                    # mw , mw_std                  = normalize(m, xp)
                    # mcw, membership_m            = k_means(mw, num_clusters=self.param.n_sub_clusters, max_iter=self.param.clstr_maxiter, thresh=self.param.clstr_threshold, rand_seed=self.param.rand_seed, xp=xp)
                    
                    # to repoduce clusters
                    xp                           = np
                    np.random.seed(self.param.rand_seed)
                    m                            = m_bte[gidx_to_pidx[grid_idx]]
                    mw , mw_std                  = normalize(m, xp)
                    
                    if mw.shape[0] >= self.param.n_sub_clusters:
                        mcw0                         = mw[np.random.choice(mw.shape[0], self.param.n_sub_clusters, replace=False)]
                    else:
                        mcw0                         = mw[np.random.choice(mw.shape[0], self.param.n_sub_clusters, replace=True)]
                    
                    mcw                          = scipy.cluster.vq.kmeans(mw, mcw0, iter=self.param.clstr_maxiter, thresh=self.param.clstr_threshold, check_finite=False)[0]
                    mcw0[0:mcw.shape[0], :]      = mcw[:,:]
                    mcw                          = mcw0
                    dist_mat                     = xp.linalg.norm(mw[:, None, :] - mcw[None, : , :], axis=2)
                    membership_m                 = xp.argmin(dist_mat, axis=1)
                    
                    assert mcw.shape[0]               == self.param.n_sub_clusters
                    mc                                = mcw * mw_std
                    self.sub_cluster_c[grid_idx]      = mc
                    self.sub_cluster_c_lbl[grid_idx]  = membership_m
                    
                    n0       = xp.ones(mc.shape[0]) * self.param.n0
                    Ex       = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey       = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    Tg       = mc[: , 2]
                    ne       = mc[: , 3] * self.param.n0
                    ni       = mc[: , 3] * self.param.n0
                    ns_by_n0 = xp.transpose(mc[: , 4:])
                    EMag     = xp.sqrt(Ex**2 + Ey**2)
                    
                    if self.param.verbose == 1 :
                        print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(self.rankG, self.npesG, grid_idx), flush=True)
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq)      , flush=True)
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx], flush=True)
                        
                        print("n0    (min)               = %.12E [1/m^3]      \t n0   (max) = %.12E [1/m^3] "%(xp.min(n0)       , xp.max(n0))   , flush=True)
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN)    , xp.max(ExbyN)), flush=True)
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN)    , xp.max(EybyN)), flush=True)
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)       , xp.max(Tg))   , flush=True)
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)       , xp.max(ne))   , flush=True)
                        
                        for i in range(ns_by_n0.shape[0]):
                            print("[%d] ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(i, xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
                                            
                    if (use_gpu==1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                            ns_by_n0 = cp.array(ns_by_n0)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    
                    # cp.save(self.param.out_fname + "_ns_by_n0_%02d.npy"%(grid_idx) , ns_by_n0 , grid_idx)
                    # cp.save(self.param.out_fname + "_n0_%02d.npy"%(grid_idx)       , n0       , grid_idx)
                    # cp.save(self.param.out_fname + "_ne_%02d.npy"%(grid_idx)       , ne       , grid_idx)
                    # cp.save(self.param.out_fname + "_ni_%02d.npy"%(grid_idx)       , ni       , grid_idx)
                    
                    # cp.save(self.param.out_fname + "_Tg_%02d.npy"%(grid_idx)  , Tg    , grid_idx)
                    # cp.save(self.param.out_fname + "_eRe_%02d.npy"%(grid_idx) , Ex    , grid_idx)
                    # cp.save(self.param.out_fname + "_eIm_%02d.npy"%(grid_idx) , Ey    , grid_idx)
                    # cp.save(self.param.out_fname + "_E_%02d.npy"%(grid_idx)   , EMag  , grid_idx)
                    
                    return
                
                with cp.cuda.Device(dev_id):
                    t1()
            
        else:
            for grid_idx in self.active_grid_idx:
                dev_id            = self.gidx_to_device_map(grid_idx, n_grids)
                
                def t1():
                    bte_idx           = gidx_to_pidx[grid_idx]
                    mc                = xp.array(m_bte[bte_idx])
                    
                    n0                = xp.ones(mc.shape[0]) * self.param.n0
                    Ex                = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey                = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    
                    Tg                = mc[: , 2]
                    ne                = mc[: , 3] * self.param.n0
                    ni                = mc[: , 3] * self.param.n0
                    ns_by_n0          = xp.transpose(mc[: , 4:])
                    EMag              = xp.sqrt(Ex**2 + Ey**2)
                
                    if self.param.verbose == 1 :
                        print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(self.rankG, self.npesG, grid_idx), flush=True)
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq)      , flush=True)
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx], flush=True)
                        
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN), xp.max(ExbyN)), flush=True)
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN), xp.max(EybyN)), flush=True)
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)   , xp.max(Tg))   , flush=True)
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)   , xp.max(ne))   , flush=True)
                        
                        for i in range(ns_by_n0.shape[0]):
                            print("ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
            
            
                    if (use_gpu == 1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                            ns_by_n0 = cp.array(ns_by_n0)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    
                    # cp.save(self.param.out_fname + "_ns_by_n0_%02d.npy"%(grid_idx) , ns_by_n0 , grid_idx)
                    # cp.save(self.param.out_fname + "_n0_%02d.npy"%(grid_idx)       , n0       , grid_idx)
                    # cp.save(self.param.out_fname + "_ne_%02d.npy"%(grid_idx)       , ne       , grid_idx)
                    # cp.save(self.param.out_fname + "_ni_%02d.npy"%(grid_idx)       , ni       , grid_idx)
                    
                    # cp.save(self.param.out_fname + "_Tg_%02d.npy"%(grid_idx)  , Tg    , grid_idx)
                    # cp.save(self.param.out_fname + "_eRe_%02d.npy"%(grid_idx) , Ex    , grid_idx)
                    # cp.save(self.param.out_fname + "_eIm_%02d.npy"%(grid_idx) , Ey    , grid_idx)
                    # cp.save(self.param.out_fname + "_E_%02d.npy"%(grid_idx)   , EMag  , grid_idx)
                    
                    return

                with cp.cuda.Device(dev_id):
                    t1()
                    
        return        

    def solve_init(self):
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        for grid_idx in range(self.param.n_grids):
            def t1():
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                print("rank [%d/%d] setting grid %d to device %d"%(rank, npes, grid_idx, dev_id), flush=True)
                self.bte_solver.host_to_device_setup(dev_id, grid_idx)
            
            t1()
        
        def ts_op_setup(grid_idx):
            xp                                      = self.xp_module 
            f_mw                                    = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
            n_pts                                   = f_mw.shape[1]
            Qmat                                    = self.bte_solver._op_qmat[grid_idx]
            INr                                     = xp.eye(Qmat.shape[1])
            self.bte_solver._op_imat_vx[grid_idx]   = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr)
            
            if self.param.use_clstr_inp==True:
                assert n_pts == self.param.n_sub_clusters
            
        if(self.param.use_gpu==1):
            self.xp_module = cp
            for grid_idx in self.active_grid_idx:
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                def t1():
                    ts_op_setup(grid_idx)
                    
                    vth          = self.bte_solver._par_vth[grid_idx]
                    qA           = self.bte_solver._op_diag_dg[grid_idx]
                    mw           = bte_utils.get_maxwellian_3d(vth, 1)
                    mm_op        = self.bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                    f_mw         = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    f_mw         = f_mw/cp.dot(mm_op, f_mw)
                    f_mw         = cp.dot(qA.T, f_mw)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", cp.copy(f_mw))
            
                with cp.cuda.Device(dev_id):
                    t1()
            
        else:
            self.xp_module = np
            for grid_idx in self.active_grid_idx:
                def t1():
                    ts_op_setup(grid_idx)
                    
                    vth          = self.bte_solver._par_vth[grid_idx]
                    qA           = self.bte_solver._op_diag_dg[grid_idx]
                    mw           = bte_utils.get_maxwellian_3d(vth, 1)
                    mm_op        = self.bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                    f_mw         = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    f_mw         = f_mw/np.dot(mm_op, f_mw)
                    f_mw         = np.dot(qA.T, f_mw)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", np.copy(f_mw))
                
                t1()
            
        return
            
    def solve_step(self, time, delta_t):
        """
        perform a single timestep in 0d-BTE
        """
        rank                    = self.rankG
        npes                    = self.npesG
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        for grid_idx in self.active_grid_idx:
            dev_id  = gidx_to_device_map(grid_idx,n_grids)

            def t1():
                
                # seting the E field for time t + dt (implicit step)
                xp    = self.bte_solver.xp_module
                eRe   = self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe")
                eIm   = self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm")
                Et    = eRe * xp.cos(2 * xp.pi * self.param.Efreq * (time + delta_t)) + eIm * xp.sin(2 * xp.pi * self.param.Efreq * (time + delta_t))
                self.bte_solver.set_boltzmann_parameter(grid_idx, "E", Et)
                
                u0    = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                v     = self.bte_solver.step(grid_idx, u0, self.param.atol, self.param.rtol, self.param.max_iter, time, delta_t)
                self.bte_solver.set_boltzmann_parameter(grid_idx, "u1", v)

            with cp.cuda.Device(dev_id):
                t1()    
        return 
    
    def solve(self):
        """
        Can be used to compute steady-state or cycle averaged BTE solutions
        """
        rank                    = self.rankG
        npes                    = self.npesG
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
        coll_list               = self.bte_solver.get_collision_list()
        coll_names              = self.bte_solver.get_collision_names()
        
        if csv_write: 
            data_csv = np.empty((self.tps_npts, 8 + len(coll_list)))
            
        for grid_idx in self.active_grid_idx:
            dev_id = gidx_to_device_map(grid_idx,n_grids)
            
            def t1():
                # print("rank [%d/%d] BTE launching grid %d on %s"%(rank, npes, grid_idx, dev_id), flush=True)
                # f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                # ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                # self.ff[grid_idx]  = ff
                # self.qoi[grid_idx] = qoi
                try:
                    print("rank [%d/%d] BTE launching grid %d on %s"%(rank, npes, grid_idx, dev_id), flush=True)
                    f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                    ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                    self.ff[grid_idx]  = ff
                    self.qoi[grid_idx] = qoi
                except Exception as e:
                    print(e)
                    print("rank [%d/%d] solver failed for v-space gird no %d"%(self.rankG, self.npesG, grid_idx), flush=True)
                    self.comm.Abort(0)
                    #sys.exit(-1)
            
            with cp.cuda.Device(dev_id):
                t1()
        return
    
    def push(self, interface):
        xp                      = self.xp_module
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        use_interp              = self.param.use_clstr_inp
        
        heavy_temp  = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts    = len(heavy_temp)
        
        n_reactions = interface.nComponents(libtps.t2bIndex.ReactionRates)
        rates       = np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False).reshape((n_reactions, tps_npts), copy=False)
        
        if (use_interp==True):
            if(n_reactions>0):
                rates[:,:] = 0.0
                for grid_idx in self.active_grid_idx:
                    dev_id = gidx_to_device_map(grid_idx,n_grids)
                    
                    def t1():
                        qA        = self.bte_solver._op_diag_dg[grid_idx]
                        u0        = self.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        h_curr    = self.bte_solver.normalized_distribution(grid_idx, u0)
                        qoi       = self.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        rr_cpu    = xp.asnumpy(qoi["rates"])
                        inp_mask  = xp.asnumpy(self.sub_cluster_c_lbl[grid_idx]) == np.arange(self.param.n_sub_clusters)[:, None]
                        
                        rr_interp = np.zeros((n_reactions, len(gidx_to_pidx_map[grid_idx])))
                        
                        for c_idx in range(self.param.n_sub_clusters):
                            inp_idx = inp_mask[c_idx]
                            for r_idx in range(n_reactions):
                                rr_interp[r_idx, inp_idx] = rr_cpu[TPSINDEX.RR_IDX[r_idx]][c_idx] * self.param.N_Avo

                        for r_idx in range(n_reactions):                            
                            rates[r_idx][gidx_to_pidx_map[grid_idx]] = rr_interp[r_idx, :]
                    
                    with cp.cuda.Device(dev_id):
                        t1()

                rates = rates.reshape((-1), copy=False)
                rates[rates<0] = 0.0

                # fname = self.param.out_fname+"_push_rank_%d_npes_%d.h5"%(self.rankG, self.npesG)
                # with h5py.File(fname, 'w') as F:
                #     F.create_dataset("Tg[K]"    , data=heavy_temp)
                #     F.create_dataset("rates"    , data=rates)
                #     F.create_dataset("rates2"   , data=np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False))
                # F.close()

        else:
            if(n_reactions>0):
                rates[:,:] = 0.0
                
                for grid_idx in self.active_grid_idx:
                    dev_id = gidx_to_device_map(grid_idx,n_grids)
                    
                    def t1():
                        qA       = self.bte_solver._op_diag_dg[grid_idx]
                        u0       = self.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        h_curr   = self.bte_solver.normalized_distribution(grid_idx, u0)
                        qoi      = self.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        rr_cpu   = xp.asnumpy(qoi["rates"])
                        
                        for r_idx in range(n_reactions):
                            rates[r_idx][gidx_to_pidx_map[grid_idx]] = rr_cpu[TPSINDEX.RR_IDX[r_idx]] * self.param.N_Avo
                        
                    with cp.cuda.Device(dev_id):
                        t1()
                        
                rates = rates.reshape((-1), copy=False)
                rates[rates<0] = 0.0
        return 
    
    async def fetch_asnyc(self, interface):
        xp                      = self.xp_module
        use_interp              = self.param.use_clstr_inp
        gidx_to_pidx            = self.grid_idx_to_spatial_idx_map
        heavy_temp              = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts                = len(heavy_temp)
        self.tps_npts           = tps_npts
        nspecies                = interface.Nspecies()
        electron_temp           = np.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False)
        efield                  = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False).reshape((2, tps_npts))
        species_densities       = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(nspecies, tps_npts)
        
        cs_avail_species        = self.bte_solver._avail_species
        
        n0                      = np.sum(species_densities, axis=0) - species_densities[TPSINDEX.ELE_IDX]
        ns_by_n0                = np.concatenate([species_densities[TPSINDEX.MOLE_FRAC_IDX[i]]/n0 for i in range(len(cs_avail_species))]).reshape((len(cs_avail_species), tps_npts))
        
        n_grids                 = self.param.n_grids 
        use_gpu                 = self.param.use_gpu
        
        Tg                      = heavy_temp
        
        Ex                      = efield[0]
        Ey                      = efield[1]
        
        EMag                    = np.sqrt(Ex**2 + Ey**2)
        e_idx                   = EMag<self.param.EMag_threshold
        
        ExbyN                   = Ex/n0/self.param.Td_fac
        EybyN                   = Ey/n0/self.param.Td_fac
        
        ExbyN[e_idx]            = (self.param.EMag_threshold/np.sqrt(2)) / n0[e_idx] / self.param.Td_fac
        EybyN[e_idx]            = (self.param.EMag_threshold/np.sqrt(2)) / n0[e_idx] / self.param.Td_fac
        
        Ex                      = ExbyN * self.param.n0 * self.param.Td_fac
        Ey                      = EybyN * self.param.n0 * self.param.Td_fac
    
        ion_deg                 = species_densities[TPSINDEX.ELE_IDX]/n0
        ion_deg[ion_deg<=0]     = 1e-16
        ns_by_n0[ns_by_n0<=0]   = 0
        m_bte                   = np.concatenate([ExbyN.reshape((-1, 1)), EybyN.reshape((-1, 1)), Tg.reshape((-1, 1)), ion_deg.reshape((-1, 1))] + [ ns_by_n0[i].reshape((-1, 1)) for i in range(ns_by_n0.shape[0])], axis=1)
        
        self.m_bte              = m_bte
        self.sub_cluster_c      = None
        self.sub_cluster_c_lbl  = None
        gidx_to_device_map      = self.gidx_to_device_map
        
        if (use_interp == True):
            n_sub_clusters               = self.param.n_sub_clusters
            self.sub_cluster_c           = [None for i in range(self.param.n_grids)]
            self.sub_cluster_c_lbl       = [None for i in range(self.param.n_grids)]
            
            def normalize(obs, xp):
                std_obs   = xp.std(obs, axis=0)
                std_obs[std_obs == 0.0] = 1.0
                return obs/std_obs, std_obs

            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                dev_id    = self.gidx_to_device_map(grid_idx, n_grids)
                @spawn(ts[grid_idx], placement=[gpu(dev_id)], vcus=0.0)
                def t1():
                    # xp                           = cp
                    # m                            = xp.array(m_bte[gidx_to_pidx[grid_idx]])
                    # mw , mw_std                  = normalize(m, xp)
                    # mcw, membership_m            = k_means(mw, num_clusters=self.param.n_sub_clusters, max_iter=self.param.clstr_maxiter, thresh=1e-8, rand_seed=self.param.rand_seed, xp=xp)
                    
                    # to repoduce clusters
                    xp                           = np
                    np.random.seed(self.param.rand_seed)
                    m                            = m_bte[gidx_to_pidx[grid_idx]]
                    mw , mw_std                  = normalize(m, xp)
                    
                    if mw.shape[0] >= self.param.n_sub_clusters:
                        mcw0                         = mw[np.random.choice(mw.shape[0], self.param.n_sub_clusters, replace=False)]
                    else:
                        mcw0                         = mw[np.random.choice(mw.shape[0], self.param.n_sub_clusters, replace=True)]
                        
                    mcw                          = scipy.cluster.vq.kmeans(mw, mcw0, iter=self.param.clstr_maxiter, thresh=self.param.clstr_threshold, check_finite=False)[0]
                    mcw0[0:mcw.shape[0], :]      = mcw[:,:]
                    mcw                          = mcw0
                    dist_mat                     = xp.linalg.norm(mw[:, None, :] - mcw[None, : , :], axis=2)
                    membership_m                 = xp.argmin(dist_mat, axis=1)
                    
                    assert mcw.shape[0]               == self.param.n_sub_clusters
                    mc                                = mcw * mw_std
                    self.sub_cluster_c[grid_idx]      = mc
                    self.sub_cluster_c_lbl[grid_idx]  = membership_m
                    
                    n0       = xp.ones(mc.shape[0]) * self.param.n0
                    Ex       = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey       = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    Tg       = mc[: , 2]
                    ne       = mc[: , 3] * self.param.n0
                    ni       = mc[: , 3] * self.param.n0
                    ns_by_n0 = xp.transpose(mc[: , 4:])
                    EMag     = xp.sqrt(Ex**2 + Ey**2)
                    
                    if self.param.verbose == 1 :
                        print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(self.rankG, self.npesG, grid_idx), flush=True)
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq)      , flush=True)
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx], flush=True)
                        
                        print("n0    (min)               = %.12E [1/m^3]      \t n0   (max) = %.12E [1/m^3] "%(xp.min(n0)       , xp.max(n0))   , flush=True)
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN)    , xp.max(ExbyN)), flush=True)
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN)    , xp.max(EybyN)), flush=True)
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)       , xp.max(Tg))   , flush=True)
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)       , xp.max(ne))   , flush=True)
                        
                        for i in range(ns_by_n0.shape[0]):
                            print("[%d] ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(i, xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
                                            
                    if (use_gpu==1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                            ns_by_n0 = cp.array(ns_by_n0)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    
                    # cp.save(self.param.out_fname + "_ns_by_n0_%02d.npy"%(grid_idx) , ns_by_n0 , grid_idx)
                    # cp.save(self.param.out_fname + "_n0_%02d.npy"%(grid_idx)       , n0       , grid_idx)
                    # cp.save(self.param.out_fname + "_ne_%02d.npy"%(grid_idx)       , ne       , grid_idx)
                    # cp.save(self.param.out_fname + "_ni_%02d.npy"%(grid_idx)       , ni       , grid_idx)
                    
                    # cp.save(self.param.out_fname + "_Tg_%02d.npy"%(grid_idx)  , Tg    , grid_idx)
                    # cp.save(self.param.out_fname + "_eRe_%02d.npy"%(grid_idx) , Ex    , grid_idx)
                    # cp.save(self.param.out_fname + "_eIm_%02d.npy"%(grid_idx) , Ey    , grid_idx)
                    # cp.save(self.param.out_fname + "_E_%02d.npy"%(grid_idx)   , EMag  , grid_idx)
                    
                    return
                
            await ts
            
        else:
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                dev_id            = self.gidx_to_device_map(grid_idx, n_grids)
                @spawn(ts[grid_idx], placement=[gpu(dev_id)], vcus=0.0)
                def t1():
                    bte_idx           = gidx_to_pidx[grid_idx]
                    mc                = xp.array(m_bte[bte_idx])
                    
                    n0                = xp.ones(mc.shape[0]) * self.param.n0
                    Ex                = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey                = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    
                    Tg                = mc[: , 2]
                    ne                = mc[: , 3] * self.param.n0
                    ni                = mc[: , 3] * self.param.n0
                    ns_by_n0          = xp.transpose(mc[: , 4:])
                    EMag              = xp.sqrt(Ex**2 + Ey**2)
                
                    if self.param.verbose == 1 :
                        print("rank [%d/%d] Boltzmann solver inputs for v-space grid id %d"%(self.rankG, self.npesG, grid_idx), flush=True)
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq)      , flush=True)
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx], flush=True)
                        
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(xp.min(ExbyN), xp.max(ExbyN)), flush=True)
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(xp.min(EybyN), xp.max(EybyN)), flush=True)
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(xp.min(Tg)   , xp.max(Tg))   , flush=True)
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(xp.min(ne)   , xp.max(ne))   , flush=True)
                        
                        for i in range(ns_by_n0.shape[0]):
                            print("ns/n0 (min)               = %.12E              \t ns/n0(max) = %.12E         "%(xp.min(ns_by_n0[i]) , xp.max(ns_by_n0[i])), flush=True)
            
            
                    if (use_gpu == 1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                            ns_by_n0 = cp.array(ns_by_n0)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0", ns_by_n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    
                    # cp.save(self.param.out_fname + "_ns_by_n0_%02d.npy"%(grid_idx) , ns_by_n0 , grid_idx)
                    # cp.save(self.param.out_fname + "_n0_%02d.npy"%(grid_idx)       , n0       , grid_idx)
                    # cp.save(self.param.out_fname + "_ne_%02d.npy"%(grid_idx)       , ne       , grid_idx)
                    # cp.save(self.param.out_fname + "_ni_%02d.npy"%(grid_idx)       , ni       , grid_idx)
                    
                    # cp.save(self.param.out_fname + "_Tg_%02d.npy"%(grid_idx)  , Tg    , grid_idx)
                    # cp.save(self.param.out_fname + "_eRe_%02d.npy"%(grid_idx) , Ex    , grid_idx)
                    # cp.save(self.param.out_fname + "_eIm_%02d.npy"%(grid_idx) , Ey    , grid_idx)
                    # cp.save(self.param.out_fname + "_E_%02d.npy"%(grid_idx)   , EMag  , grid_idx)
                    
                    return
            await ts
        return        

    async def solve_init_async(self):
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        ts = TaskSpace("T")
        for grid_idx in range(self.param.n_grids):
            @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
            def t1():
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                print("rank [%d/%d] setting grid %d to device %d"%(rank, npes, grid_idx, dev_id), flush=True)
                self.bte_solver.host_to_device_setup(dev_id, grid_idx)
            
        await ts
        
        def ts_op_setup(grid_idx):
            xp                                      = self.xp_module 
            f_mw                                    = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
            n_pts                                   = f_mw.shape[1]
            Qmat                                    = self.bte_solver._op_qmat[grid_idx]
            INr                                     = xp.eye(Qmat.shape[1])
            self.bte_solver._op_imat_vx[grid_idx]   = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr)
            
            if self.param.use_clstr_inp==True:
                assert n_pts == self.param.n_sub_clusters
            
        if(self.param.use_gpu==1):
            self.xp_module = cp
            
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                @spawn(ts[grid_idx], placement=[gpu(dev_id)], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    
                    vth          = self.bte_solver._par_vth[grid_idx]
                    qA           = self.bte_solver._op_diag_dg[grid_idx]
                    mw           = bte_utils.get_maxwellian_3d(vth, 1)
                    mm_op        = self.bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                    f_mw         = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    f_mw         = f_mw/cp.dot(mm_op, f_mw)
                    f_mw         = cp.dot(qA.T, f_mw)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", cp.copy(f_mw))
            await ts
        else:
            self.xp_module = np
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    
                    vth          = self.bte_solver._par_vth[grid_idx]
                    qA           = self.bte_solver._op_diag_dg[grid_idx]
                    mw           = bte_utils.get_maxwellian_3d(vth, 1)
                    mm_op        = self.bte_solver._op_mass[grid_idx] * mw(0) * vth**3
                    f_mw         = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    f_mw         = f_mw/np.dot(mm_op, f_mw)
                    f_mw         = np.dot(qA.T, f_mw)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", np.copy(f_mw))
                    
            await ts
        
        return
            
    async def solve_step_async(self, time, delta_t):
        """
        perform a single timestep in 0d-BTE
        """
        rank                    = self.rankG
        npes                    = self.npesG
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        
        ts = TaskSpace("T")
        for grid_idx in self.active_grid_idx:
            @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
            def t1():
                
                # seting the E field for time t + dt (implicit step)
                xp    = self.bte_solver.xp_module
                eRe   = self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe")
                eIm   = self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm")
                Et    = eRe * xp.cos(2 * xp.pi * self.param.Efreq * (time + delta_t)) + eIm * xp.sin(2 * xp.pi * self.param.Efreq * (time + delta_t))
                self.bte_solver.set_boltzmann_parameter(grid_idx, "E", Et)
                
                u0    = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                v     = self.bte_solver.step(grid_idx, u0, self.param.atol, self.param.rtol, self.param.max_iter, time, delta_t)
                self.bte_solver.set_boltzmann_parameter(grid_idx, "u1", v)
                
        await ts
        
        return 
    
    async def solve_async(self):
        """
        Can be used to compute steady-state or cycle averaged BTE solutions
        """
        rank                    = self.rankG
        npes                    = self.npesG
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
        assert num_gpus         == self.num_gpus_per_node, "CuPy and Parla number of GPUs per node does not match %d vs. %d"%(num_gpus, self.num_gpus_per_node)
        coll_list               = self.bte_solver.get_collision_list()
        coll_names              = self.bte_solver.get_collision_names()
        
        if (use_gpu==1):
            parla_placement = [gpu(gidx_to_device_map(grid_idx,n_grids)) for grid_idx in range(n_grids)]
        else:
            parla_placement = [cpu for grid_idx in range(n_grids)]

        if csv_write: 
            data_csv = np.empty((self.tps_npts, 8 + len(coll_list)))
            
        ts = TaskSpace("T")
        for grid_idx in self.active_grid_idx:
            @spawn(ts[grid_idx], placement=[parla_placement[grid_idx]], vcus=0.0)
            def t1():
                try:
                    #cp.cuda.nvtx.RangePush("bte_solve")
                    print("rank [%d/%d] BTE launching grid %d on %s"%(rank, npes, grid_idx, parla_placement[grid_idx]), flush=True)
                    f0 = self.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                    ff , qoi = self.bte_solver.solve(grid_idx, f0, self.param.atol, self.param.rtol, self.param.max_iter, self.param.solver_type)
                    self.ff[grid_idx]  = ff
                    self.qoi[grid_idx] = qoi
                    #cp.cuda.nvtx.RangePop()
                except Exception as e:
                    print("rank [%d/%d] solver failed for v-space gird no %d with error = %s"%(self.rankG, self.npesG, grid_idx, str(e)), flush=True)
                    sys.exit(-1)
                    
        await ts
        return
    
    async def push_async(self, interface):
        xp                      = self.xp_module
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        use_interp              = self.param.use_clstr_inp
        
        heavy_temp  = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts    = len(heavy_temp)
        
        n_reactions = interface.nComponents(libtps.t2bIndex.ReactionRates)
        rates       = np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False).reshape((n_reactions, tps_npts), copy=False)
        
        if (use_interp==True):
            if(n_reactions>0):
                rates[:,:] = 0.0
                ts = TaskSpace("T")
                for grid_idx in self.active_grid_idx:
                    @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                    def t1():
                        qA        = self.bte_solver._op_diag_dg[grid_idx]
                        u0        = self.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        h_curr    = self.bte_solver.normalized_distribution(grid_idx, u0)
                        qoi       = self.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        rr_cpu    = xp.asnumpy(qoi["rates"])
                        inp_mask  = xp.asnumpy(self.sub_cluster_c_lbl[grid_idx]) == np.arange(self.param.n_sub_clusters)[:, None]
                        
                        rr_interp = np.zeros((n_reactions, len(gidx_to_pidx_map[grid_idx])))
                        
                        for c_idx in range(self.param.n_sub_clusters):
                            inp_idx = inp_mask[c_idx]
                            for r_idx in range(n_reactions):
                                rr_interp[r_idx, inp_idx] = rr_cpu[TPSINDEX.RR_IDX[r_idx]][c_idx] * self.param.N_Avo

                        for r_idx in range(n_reactions):                            
                            rates[r_idx][gidx_to_pidx_map[grid_idx]] = rr_interp[r_idx, :]
                await ts
                rates = rates.reshape((-1))
                rates[rates<0] = 0.0
        else:
            if(n_reactions>0):
                rates[:,:] = 0.0
                ts = TaskSpace("T")
                for grid_idx in self.active_grid_idx:
                    @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                    def t1():
                        qA       = self.bte_solver._op_diag_dg[grid_idx]
                        u0       = self.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        h_curr   = self.bte_solver.normalized_distribution(grid_idx, u0)
                        qoi      = self.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        rr_cpu   = xp.asnumpy(qoi["rates"])
                        
                        for r_idx in range(n_reactions):
                            rates[r_idx][gidx_to_pidx_map[grid_idx]] = rr_cpu[TPSINDEX.RR_IDX[r_idx]] * self.param.N_Avo
                        
                await ts
                rates = rates.reshape((-1))
                rates[rates<0] = 0.0
        return 
    
    def io_output_data(self, grid_idx, u0, plot_data:bool, export_csv:bool, fname:str):
        xp                      = self.xp_module
        gidx_to_device_map      = self.gidx_to_device_map
        n_grids                 = self.param.n_grids
        dev_id                  = gidx_to_device_map(grid_idx, n_grids)
        qA                      = self.bte_solver._op_diag_dg[grid_idx]
        h_curr                  = xp.dot(qA, u0)
        h_curr                  = self.bte_solver.normalized_distribution(grid_idx, h_curr)
        ff                      = h_curr
        qoi                     = self.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
        coll_list               = self.bte_solver.get_collision_list()
        coll_names              = self.bte_solver.get_collision_names()
        cs_data                 = self.bte_solver.get_cross_section_data()
        
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
        ev       = np.linspace(1e-3, self.bte_solver._par_ev_range[grid_idx][1], 500)
        ff_r     = self.bte_solver.compute_radial_components(grid_idx, ev, ff_cpu)
        
        n0       = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "n0"))
        ne       = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ne"))
        ni       = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ni"))
        ns_by_n0 = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "ns_by_n0")).T
        Tg       = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "Tg"))
        eRe      = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe"))
        eIm      = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm"))
        eMag     = np.sqrt(eRe**2 + eIm**2)
        
        data_csv = np.zeros((ne.shape[0], 7 + ns_by_n0.shape[1] + len((coll_list)) + 2))    

        if export_csv:
            data_csv[: , 0]    = n0
            data_csv[: , 1]    = ne/n0
            idx                =2 + ns_by_n0.shape[1]
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
                
            with open("%s_qoi.csv"%(fname), 'w', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter=',')
                # write the header
                header = ["n0", "ne/n0"] + ["(%s)/n0"%(s) for s in cs_species] + ["Tg", "eRe", "eIm", "E",  "energy", "mobility", "diffusion"]
                for col_idx, g in enumerate(coll_list):
                    header.append(str(coll_names[col_idx]))
                
                writer.writerow(header)
                writer.writerows(data_csv)

        if plot_data:
            n_pts        = ff_cpu.shape[1]
            num_sh       = len(self.bte_solver._par_lm[grid_idx])
            num_subplots = num_sh 
            num_plt_cols = min(num_sh, 4)
            num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
            fig          = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=200, constrained_layout=True)
            plt_idx      =  1
            n_pts_step   =  n_pts // 20

            for lm_idx, lm in enumerate(self.bte_solver._par_lm[grid_idx]):
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
            
        return        

    def params_dump(self):
        params_dict = dict()
        params_dict["sp_order"]         = self.param.sp_order
        params_dict["spline_qpts"]      = self.param.spline_qpts
        params_dict["Nr"]               = self.param.Nr
        params_dict["l_max"]            = self.param.l_max
        params_dict["ev_max"]           = self.param.ev_max
        params_dict["n_grids"]          = self.param.n_grids
        params_dict["n_sub_clusters"]   = self.param.n_sub_clusters
        params_dict["dt"]               = self.param.dt
        params_dict["cycles"]           = self.param.cycles
        params_dict["solver_type"]      = self.param.solver_type
        params_dict["atol"]             = self.param.atol
        params_dict["rtol"]             = self.param.rtol
        params_dict["max_iter"]         = self.param.max_iter
        params_dict["tps_bte_max_iter"] = self.param.tps_bte_max_iter
        params_dict["bte_solve_freq"]   = self.param.bte_solve_freq
        params_dict["ee_collisions"]    = self.param.ee_collisions
        params_dict["use_gpu"]          = self.param.use_gpu
        params_dict["dev_id"]           = self.param.dev_id
        params_dict["collisions"]       = self.param.collisions
        params_dict["export_csv"]       = self.param.export_csv
        params_dict["plot_data"]        = self.param.plot_data
        params_dict["Efreq"]            = self.param.Efreq
        params_dict["verbose"]          = self.param.verbose
        params_dict["Te"]               = self.param.Te
        params_dict["threads"]          = self.param.threads
        params_dict["grid_idx"]         = self.param.grid_idx
        params_dict["output_dir"]       = self.param.output_dir
        params_dict["out_fname"]        = self.param.out_fname
        params_dict["rand_seed"]        = self.param.rand_seed
        params_dict["use_clstr_inp"]    = self.param.use_clstr_inp
        
        return params_dict
        
    
class pp(enum.IntEnum):
    BTE_SETUP     = 0
    BTE_FETCH     = 1
    BTE_SOLVE     = 2
    BTE_PUSH      = 3
    TPS_SETUP     = 4
    TPS_FETCH     = 5
    TPS_SOLVE     = 6
    TPS_PUSH      = 7
    LAST          = 8
    
profile_nn     = ["bte_setup", "bte_fetch", "bte_solve", "bte_push", "tps_setup", "tps_fetch", "tps_solve", "tps_push", "last"]
profile_tt     = [profile_t(profile_nn[i]) for i in range(int(pp.LAST))]

def profile_stats(boltzmann:Boltzmann0D2VBactchedSolver, p_tt: profile_t, p_nn, fname, comm):
    
    rank = comm.Get_rank()
    npes = comm.Get_size()
    
    Nx = boltzmann.param.n_grids * boltzmann.param.n_sub_clusters
    Nv = (boltzmann.param.Nr + 1) * (boltzmann.param.l_max + 1)
    
    tt = list()
    for i in range(len(p_tt)):
        tt.append(min_mean_max(p_tt[i].seconds/p_tt[i].iter, comm))
        
    header = [  "Nv",
                "Nx",
                "bte_setup_min", "bte_setup_mean", "bte_setup_max",
                "bte_fetch_min", "bte_fetch_mean", "bte_fetch_max",
                "bte_solve_min", "bte_solve_mean", "bte_solve_max",
                "bte_push_min" , "bte_push_mean" , "bte_push_max",
                
                "tps_setup_min", "tps_setup_mean", "tps_setup_max",
                "tps_fetch_min", "tps_fetch_mean", "tps_fetch_max",
                "tps_solve_min", "tps_solve_mean", "tps_solve_max",
                "tps_push_min" , "tps_push_mean" , "tps_push_max"]
    
    data   = [  Nv,
                Nx,
                tt[pp.BTE_SETUP][0], tt[pp.BTE_SETUP][1], tt[pp.BTE_SETUP][2], 
                tt[pp.BTE_FETCH][0], tt[pp.BTE_FETCH][1], tt[pp.BTE_FETCH][2], 
                tt[pp.BTE_SOLVE][0], tt[pp.BTE_SOLVE][1], tt[pp.BTE_SOLVE][2], 
                tt[pp.BTE_PUSH][0] , tt[pp.BTE_PUSH][1] , tt[pp.BTE_PUSH][2] ,
                tt[pp.TPS_SETUP][0], tt[pp.TPS_SETUP][1], tt[pp.TPS_SETUP][2], 
                tt[pp.TPS_FETCH][0], tt[pp.TPS_FETCH][1], tt[pp.TPS_FETCH][2], 
                tt[pp.TPS_SOLVE][0], tt[pp.TPS_SOLVE][1], tt[pp.TPS_SOLVE][2], 
                tt[pp.TPS_PUSH][0] , tt[pp.TPS_PUSH][1] , tt[pp.TPS_PUSH][2] ]
    
    data_str= ["%.4E"%d for d in data]
    
    if rank ==0 :
        if fname!="":
            with open(fname, "a") as f:
                f.write("nprocs: %d timestamp %s \n"%(npes, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
                f.write(""+str(boltzmann.params_dump())+"\n")
                f.write(",".join(header)+"\n")
                f.write(",".join(data_str)+"\n")
                f.write("---" + "\n")
                f.close()
        else:
            print(",".join(header))
            print(",".join(data_str))


def driver_w_parla(comm):
    
    rank = comm.Get_rank()
    npes = comm.Get_size()
    
    with Parla():
        dev_id = rank % len(gpu)
        @spawn(placement=[gpu(dev_id)], vcus=0)
        async def __main__():
            # TPS solver
            profile_tt[pp.TPS_SETUP].start()
            tps = libtps.Tps(comm)
            tps.parseCommandLineArgs(sys.argv)
            tps.parseInput()
            tps.chooseDevices()
            tps.chooseSolver()
            tps.initialize()
            profile_tt[pp.TPS_SETUP].stop()
            
            interface = libtps.Tps2Boltzmann(tps)
            tps.initInterface(interface)
            tps.solveBegin()
            # --- first TPS step is needed to initialize the EM fields
            tps.solveStep()
            tps.push(interface)
            
            boltzmann = Boltzmann0D2VBactchedSolver(tps, comm)
            rank      = boltzmann.comm.Get_rank()
            npes      = boltzmann.comm.Get_size()
            
            profile_tt[pp.BTE_SETUP].start()
            boltzmann.grid_setup(interface)
            profile_tt[pp.BTE_SETUP].stop()
            
            await boltzmann.solve_init_async()
            xp        = boltzmann.bte_solver.xp_module
            max_iters = boltzmann.param.tps_bte_max_iter
            iter      = 0
            tt        = 0
            tau       = (1/boltzmann.param.Efreq)
            dt_tps    = interface.timeStep()
            dt_bte    = boltzmann.param.dt * tau 
            bte_steps = int(dt_tps/dt_bte)
            n_grids   = boltzmann.param.n_grids
            
            cycle_freq           = 1 #int(xp.ceil(tau/dt_tps))
            terminal_output_freq = -1
            gidx_to_device_map = boltzmann.gidx_to_device_map
            
            tps_sper_cycle = int(xp.ceil(tau/dt_tps))
            bte_sper_cycle = int(xp.ceil(tau/dt_bte))
            bte_max_cycles = int(boltzmann.param.cycles)
            tps_max_cycles = boltzmann.param.bte_solve_freq
            
            if (boltzmann.rankG==0):
                print("tps steps per cycle : ", tps_sper_cycle, "bte_steps per cycle", bte_sper_cycle, flush=True)
                
            while (iter<max_iters):
                if (iter%cycle_freq==0):
                    interface.saveDataCollection(cycle=(iter//cycle_freq), time=iter)
                
                # ########################## BTE solve ##################################################
                profile_tt[pp.BTE_FETCH].start()
                await boltzmann.fetch_asnyc(interface)
                profile_tt[pp.BTE_FETCH].stop()
                
                if (boltzmann.param.solver_type=="steady-state"):
                    
                    profile_tt[pp.BTE_SOLVE].start()
                    await boltzmann.solve_async()
                    profile_tt[pp.BTE_SOLVE].stop()
                    
                    profile_tt[pp.BTE_PUSH].start()
                    ts = TaskSpace("T")
                    for grid_idx in boltzmann.active_grid_idx:
                        @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                        def t1():
                            boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", boltzmann.ff[grid_idx])
                    await ts
                    await boltzmann.push_async(interface)
                    profile_tt[pp.BTE_PUSH].stop()
                    
                    if boltzmann.param.export_csv ==1:
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id  = gidx_to_device_map(grid_idx,n_grids)
                            with cp.cuda.Device(dev_id):
                                u_vec   = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                                boltzmann.io_output_data(grid_idx, u_vec, plot_data=True, export_csv=True, fname=boltzmann.param.out_fname+"_grid_%02d_rank_%d_npes_%d"%(grid_idx, rank, npes))
                else:
                    assert boltzmann.param.solver_type == "transient", "unknown BTE solver type"
                    
                    tt_bte       = 0
                    bte_u        = [0 for i in range(n_grids)]
                    bte_v        = [0 for i in range(n_grids)]
                    
                    u_avg        = [0 for i in range(n_grids)]
                    
                    abs_error    = [0 for i in range(n_grids)]
                    rel_error    = [0 for i in range(n_grids)]
                    cycle_f1     = (0.5 * dt_bte/ (bte_sper_cycle * dt_bte))
                    
                    for bte_idx in range(bte_sper_cycle * bte_max_cycles +1):
                        if (bte_idx % bte_sper_cycle == 0):
                            ts = TaskSpace("T")
                            for grid_idx in boltzmann.active_grid_idx:
                                @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                                def t1():
                                    u0      = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                                    fname   = "%s_iter%04d_grid_%04d_cycle_%0d"%(boltzmann.param.out_fname, iter, grid_idx, bte_idx//bte_sper_cycle)
                                    
                                    abs_error[grid_idx] = xp.max(xp.abs(bte_v[grid_idx]-u0))
                                    rel_error[grid_idx] = abs_error[grid_idx] / xp.max(xp.abs(u0))
                                    #print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E"%(bte_idx, tt_bte, abs_error[grid_idx], rel_error[grid_idx]))
                                    
                                    # if(bte_idx >0):
                                    #     print(grid_idx, " u_ptr ", u_avg[grid_idx].data, " v_ptr " , v_avg[grid_idx].data)
                                    
                                    bte_v[grid_idx] = xp.copy(u0)
                                    
                            await ts
                            p_t3 = min_mean_max(profile_tt[pp.BTE_SOLVE].snap, comm)
                            print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E --- runtime = %.4E (s) "%(bte_idx, tt_bte, max(abs_error), max(rel_error), p_t3[2]), flush=True)
                            
                            if max(abs_error) < boltzmann.param.atol or max(rel_error)< boltzmann.param.rtol:
                                break
                            
                            if bte_idx < bte_sper_cycle * bte_max_cycles:
                                u_avg  = [0 for i in range(n_grids)]
                                
                        if bte_idx == bte_sper_cycle * bte_max_cycles :
                            break    
                            
                        ts = TaskSpace("T")
                        for grid_idx in boltzmann.active_grid_idx:
                            @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                            def t1():
                                u_avg[grid_idx] += cycle_f1 * boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                        await ts
                        
                        profile_tt[pp.BTE_SOLVE].start()
                        await boltzmann.solve_step_async(tt_bte, dt_bte)
                        profile_tt[pp.BTE_SOLVE].stop()
                        
                        if(terminal_output_freq > 0 and bte_idx % terminal_output_freq ==0):
                            p_t3 = min_mean_max(profile_tt[pp.BTE_SOLVE].snap, comm)
                            print("[BTE] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (bte_idx, tt_bte, p_t3[0], p_t3[1], p_t3[2]))
                        
                        ts = TaskSpace("T")
                        for grid_idx in boltzmann.active_grid_idx:
                            @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                            def t1():
                                u_avg[grid_idx] += cycle_f1 * boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1")
                                boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u0", boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1"))
                        await ts
                    
                        tt_bte += dt_bte
                    
                    profile_tt[pp.BTE_PUSH].start()
                    ts = TaskSpace("T")
                    for grid_idx in boltzmann.active_grid_idx:
                        @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                        def t1():
                            xp               = boltzmann.xp_module
                            qA               = boltzmann.bte_solver._op_diag_dg[grid_idx]
                            u_avg[grid_idx]  = xp.dot(qA, u_avg[grid_idx])
                            boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", u_avg[grid_idx])
                    await ts
                    await boltzmann.push_async(interface)
                    profile_tt[pp.BTE_PUSH].stop()
                    
                    if boltzmann.param.export_csv ==1:
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id  = gidx_to_device_map(grid_idx,n_grids)
                            with cp.cuda.Device(dev_id):
                                u_vec   = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                                boltzmann.io_output_data(grid_idx, u_vec, plot_data=True, export_csv=True, fname=boltzmann.param.out_fname+"_grid_%02d_rank_%d_npes_%d"%(grid_idx, rank, npes))
                
                ################### tps solve ######################################
                profile_tt[pp.TPS_FETCH].start()
                tps.fetch(interface)
                profile_tt[pp.TPS_FETCH].stop()
                
                tps_u  = 0
                tps_v  = 0
                tt_tps = 0
                for tps_idx in range(tps_sper_cycle * tps_max_cycles + 1):
                    if (tps_idx % tps_sper_cycle == 0):
                        tps.push(interface)
                        nspecies            = interface.Nspecies()
                        heavy_temp          = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
                        tps_npts            = len(heavy_temp)
                        tps_u               = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(nspecies, tps_npts)
                        # rates               = np.array(interface.HostRead(libtps.t2bIndex.ReactionRates), copy=False).reshape((1, tps_npts))
                        # print("rates", np.min(rates[0]), np.max(rates[0]))
                        
                        abs_error           = np.linalg.norm(tps_u - tps_v, axis=1)
                        rel_error           = abs_error / np.linalg.norm(tps_u, axis=1)
                        tps_v               = np.copy(tps_u)
                        
                        p_t3 = min_mean_max(profile_tt[pp.TPS_SOLVE].snap, comm)
                        print("[TPS] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E -- runtime = %.4E (s)"%(tps_idx, tt_tps, np.max(abs_error), np.max(rel_error), p_t3[2]), flush=True)
                        # if (np.max(abs_error) < boltzmann.param.atol or np.max(rel_error) < max(1e-6,boltzmann.param.rtol)):
                        #     break
                    
                    if (tps_idx == tps_sper_cycle * tps_max_cycles):
                        break
                    
                    profile_tt[pp.TPS_SOLVE].start()
                    tps.solveStep()
                    profile_tt[pp.TPS_SOLVE].stop()
                    if(terminal_output_freq > 0 and tps_idx % terminal_output_freq ==0):
                        p_t3 = min_mean_max(profile_tt[pp.TPS_SOLVE].snap, comm)
                        print("[TPS] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (tps_idx,tt_tps, p_t3[0],p_t3[1],p_t3[2]), flush=True)
                    tt_tps +=dt_tps
                
                profile_tt[pp.TPS_PUSH].start()
                tps.push(interface)
                profile_tt[pp.TPS_PUSH].stop()
                
                tt += dt_tps * tps_idx
                iter+=1
            
            profile_stats(boltzmann, profile_tt, profile_nn, boltzmann.param.out_fname+"_profile.csv" , comm)
            tps.solveEnd()
            comm.Barrier()
            return tps.getStatus()

def driver_wo_parla(comm):
    
    rank = comm.Get_rank()
    npes = comm.Get_size()

    dev_id = rank % (cp.cuda.runtime.getDeviceCount())
    
    
    with cp.cuda.Device(dev_id):
        def __main__():
            # TPS solver
            profile_tt[pp.TPS_SETUP].start()
            tps = libtps.Tps(comm)
            tps.parseCommandLineArgs(sys.argv)
            tps.parseInput()
            tps.chooseDevices()
            tps.chooseSolver()
            tps.initialize()
            profile_tt[pp.TPS_SETUP].stop()
            
            interface = libtps.Tps2Boltzmann(tps)
            tps.initInterface(interface)
            tps.solveBegin()
            # --- first TPS step is needed to initialize the EM fields
            tps.solveStep()
            tps.push(interface)
            
            boltzmann = Boltzmann0D2VBactchedSolver(tps, comm)
            rank      = boltzmann.comm.Get_rank()
            npes      = boltzmann.comm.Get_size()
            
            profile_tt[pp.BTE_SETUP].start()
            boltzmann.grid_setup(interface)
            profile_tt[pp.BTE_SETUP].stop()
            
            boltzmann.solve_init()
            xp        = boltzmann.bte_solver.xp_module
            max_iters = boltzmann.param.tps_bte_max_iter
            iter      = 0
            tt        = 0
            tau       = (1/boltzmann.param.Efreq)
            dt_tps    = interface.timeStep()
            dt_bte    = boltzmann.param.dt * tau 
            bte_steps = int(dt_tps/dt_bte)
            n_grids   = boltzmann.param.n_grids
            
            cycle_freq           = 1 #int(xp.ceil(tau/dt_tps))
            terminal_output_freq = -1
            gidx_to_device_map = boltzmann.gidx_to_device_map
            
            tps_sper_cycle = int(xp.ceil(tau/dt_tps))
            bte_sper_cycle = int(xp.ceil(tau/dt_bte))
            bte_max_cycles = int(boltzmann.param.cycles)
            tps_max_cycles = boltzmann.param.bte_solve_freq
            
            if (boltzmann.rankG==0):
                print("tps steps per cycle : ", tps_sper_cycle, "bte_steps per cycle", bte_sper_cycle, flush=True)
                
            while (iter<max_iters):
                # if (iter%cycle_freq==0):
                #     interface.saveDataCollection(cycle=(iter//cycle_freq), time=iter)
                
                # ########################## BTE solve ##################################################
                profile_tt[pp.BTE_FETCH].start()
                boltzmann.fetch(interface)
                profile_tt[pp.BTE_FETCH].stop()
                
                if (boltzmann.param.solver_type=="steady-state"):
                    
                    profile_tt[pp.BTE_SOLVE].start()
                    boltzmann.solve()
                    profile_tt[pp.BTE_SOLVE].stop()
                    
                    profile_tt[pp.BTE_PUSH].start()
                    
                    for grid_idx in boltzmann.active_grid_idx:
                        def t1():
                            boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", boltzmann.ff[grid_idx])
                        t1()
                        
                    boltzmann.push(interface)
                    profile_tt[pp.BTE_PUSH].stop()
                    
                    if boltzmann.param.export_csv ==1:
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id  = gidx_to_device_map(grid_idx,n_grids)
                            with cp.cuda.Device(dev_id):
                                u_vec   = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                                boltzmann.io_output_data(grid_idx, u_vec, plot_data=True, export_csv=True, fname=boltzmann.param.out_fname+"_grid_%02d_rank_%d_npes_%d"%(grid_idx, rank, npes))
                
                else:
                    # boltzmann.param.solver_type = "steady-state"
                    # boltzmann.solve()
                    # for grid_idx in boltzmann.active_grid_idx:
                    #     def t1():
                    #         boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u0", boltzmann.ff[grid_idx])
                    #     t1()
                    # boltzmann.param.solver_type = "transient"
                    assert boltzmann.param.solver_type == "transient", "unknown BTE solver type"
                    tt_bte       = 0
                    bte_u        = [0 for i in range(n_grids)]
                    bte_v        = [0 for i in range(n_grids)]
                    
                    u_avg        = [0 for i in range(n_grids)]
                    
                    abs_error    = [0 for i in range(n_grids)]
                    rel_error    = [0 for i in range(n_grids)]
                    cycle_f1     = (0.5 * dt_bte/ (bte_sper_cycle * dt_bte))
                    
                    for bte_idx in range(bte_sper_cycle * bte_max_cycles +1):
                        if (bte_idx % bte_sper_cycle == 0):
                            for grid_idx in boltzmann.active_grid_idx:
                                dev_id = gidx_to_device_map(grid_idx,n_grids)
                                def t1():
                                    u0      = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                                    fname   = "%s_iter%04d_grid_%04d_cycle_%0d"%(boltzmann.param.out_fname, iter, grid_idx, bte_idx//bte_sper_cycle)
                                    
                                    abs_error[grid_idx] = xp.max(xp.abs(bte_v[grid_idx]-u0))
                                    rel_error[grid_idx] = abs_error[grid_idx] / xp.max(xp.abs(u0))
                                    #print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E"%(bte_idx, tt_bte, abs_error[grid_idx], rel_error[grid_idx]))
                                    
                                    # if(bte_idx >0):
                                    #     print(grid_idx, " u_ptr ", u_avg[grid_idx].data, " v_ptr " , v_avg[grid_idx].data)
                                    
                                    bte_v[grid_idx] = xp.copy(u0)
                                    
                                with cp.cuda.Device(dev_id):
                                    t1()

                            p_t3 = min_mean_max(profile_tt[pp.BTE_SOLVE].snap, comm)
                            print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E --- runtime = %.4E (s) "%(bte_idx, tt_bte, max(abs_error), max(rel_error), p_t3[2]), flush=True)
                            
                            if max(abs_error) < boltzmann.param.atol or max(rel_error)< boltzmann.param.rtol:
                                break
                            
                            if bte_idx < bte_sper_cycle * bte_max_cycles:
                                u_avg  = [0 for i in range(n_grids)]
                                
                        if bte_idx == bte_sper_cycle * bte_max_cycles :
                            break    
                            
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id = gidx_to_device_map(grid_idx,n_grids)
                            def t1():
                                u_avg[grid_idx] += cycle_f1 * boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                                
                            with cp.cuda.Device(dev_id):
                                t1()
                        
                        profile_tt[pp.BTE_SOLVE].start()
                        boltzmann.solve_step(tt_bte, dt_bte)
                        profile_tt[pp.BTE_SOLVE].stop()
                        
                        if(terminal_output_freq > 0 and bte_idx % terminal_output_freq ==0):
                            p_t3 = min_mean_max(profile_tt[pp.BTE_SOLVE].snap, comm)
                            print("[BTE] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (bte_idx, tt_bte, p_t3[0], p_t3[1], p_t3[2]))
                        
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id = gidx_to_device_map(grid_idx,n_grids)
                            def t1():
                                u_avg[grid_idx] += cycle_f1 * boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1")
                                boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u0", boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1"))
                            
                            with cp.cuda.Device(dev_id):
                                t1()
                        
                        tt_bte += dt_bte
                    
                    profile_tt[pp.BTE_PUSH].start()
                    for grid_idx in boltzmann.active_grid_idx:
                        dev_id = gidx_to_device_map(grid_idx,n_grids)
                        
                        def t1():
                            xp               = boltzmann.xp_module
                            qA               = boltzmann.bte_solver._op_diag_dg[grid_idx]
                            u_avg[grid_idx]  = xp.dot(qA, u_avg[grid_idx])
                            boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", u_avg[grid_idx])
                        
                        with cp.cuda.Device(dev_id):
                            t1()
                    
                    boltzmann.push(interface)
                    profile_tt[pp.BTE_PUSH].stop()
                    
                    if boltzmann.param.export_csv ==1:
                        for grid_idx in boltzmann.active_grid_idx:
                            dev_id  = gidx_to_device_map(grid_idx,n_grids)
                            with cp.cuda.Device(dev_id):
                                u_vec   = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                                boltzmann.io_output_data(grid_idx, u_vec, plot_data=True, export_csv=True, fname=boltzmann.param.out_fname+"_grid_%02d_rank_%d_npes_%d"%(grid_idx, rank, npes))
                
                ################### tps solve ######################################
                profile_tt[pp.TPS_FETCH].start()
                tps.fetch(interface)
                profile_tt[pp.TPS_FETCH].stop()

                if (iter%cycle_freq==0):
                    interface.saveDataCollection(cycle=(iter//cycle_freq), time=iter)
                
                tps_u  = 0
                tps_v  = 0
                tt_tps = 0
                for tps_idx in range(tps_sper_cycle * tps_max_cycles + 1):
                    if (tps_idx % tps_sper_cycle == 0):
                        tps.push(interface)
                        nspecies            = interface.Nspecies()
                        heavy_temp          = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
                        tps_npts            = len(heavy_temp)
                        tps_u               = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(nspecies, tps_npts)
                        # rates               = np.array(interface.HostRead(libtps.t2bIndex.ReactionRates), copy=False).reshape((1, tps_npts))
                        # print("rates", np.min(rates[0]), np.max(rates[0]))
                        
                        abs_error           = np.linalg.norm(tps_u - tps_v, axis=1)
                        rel_error           = abs_error / np.linalg.norm(tps_u, axis=1)
                        tps_v               = np.copy(tps_u)
                        
                        p_t3 = min_mean_max(profile_tt[pp.TPS_SOLVE].snap, comm)
                        print("[TPS] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E -- runtime = %.4E (s)"%(tps_idx, tt_tps, np.max(abs_error), np.max(rel_error), p_t3[2]), flush=True)
                        # if (np.max(abs_error) < boltzmann.param.atol or np.max(rel_error) < max(1e-6,boltzmann.param.rtol)):
                        #     break
                    
                    if (tps_idx == tps_sper_cycle * tps_max_cycles):
                        break
                    
                    profile_tt[pp.TPS_SOLVE].start()
                    tps.solveStep()
                    profile_tt[pp.TPS_SOLVE].stop()
                    if(terminal_output_freq > 0 and tps_idx % terminal_output_freq ==0):
                        p_t3 = min_mean_max(profile_tt[pp.TPS_SOLVE].snap, comm)
                        print("[TPS] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (tps_idx,tt_tps, p_t3[0],p_t3[1],p_t3[2]), flush=True)
                    tt_tps +=dt_tps
                
                profile_tt[pp.TPS_PUSH].start()
                tps.push(interface)
                profile_tt[pp.TPS_PUSH].stop()
                
                tt += dt_tps * tps_idx
                iter+=1
            
            profile_stats(boltzmann, profile_tt, profile_nn, boltzmann.param.out_fname+"_profile.csv" , comm)
            tps.solveEnd()
            comm.Barrier()
            return tps.getStatus()

        __main__()

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    print("running without parla")
    driver_wo_parla(comm)
    
    # print("running with parla")
    # driver_w_parla(comm)
    
            
            
