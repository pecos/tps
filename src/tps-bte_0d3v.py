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
import scipy.cluster
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
    n_sub_clusters= 300         # number of sub-clusters

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
    N_Avo         = scipy.constants.Avogadro
    
    n0            = 3.22e22 #[m^{-3}]
    
    
class TPSINDEX():
    """
    simple index map to differnt fields, from the TPS arrays
    """
    ION_IDX = 0                         # ion      density index
    ELE_IDX = 1                         # electron density index
    NEU_IDX = 2                         # neutral  density index
    
    EF_RE_IDX = 0                       # Re(E) index
    EF_IM_IDX = 1                       # Im(E) index
    
    # in future we need to setup this methodically
    # here key denotes the idx running from 0, nreactions-1
    # value denotes the reaction index in the qoi array
    RR_IDX   = {0:1}                    
    
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
        
        xp            = self.xp_module
        n_grids       = self.param.n_grids
        Te            = xp.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False) / self.param.ev_to_K # [eV]
        
        Tew           = scipy.cluster.vq.whiten(Te)
        Tecw          = scipy.cluster.vq.kmeans(Tew, np.linspace(np.min(Tew), np.max(Tew), n_grids), iter=1000, thresh=1e-8, check_finite=False)[0]
        Te_b          = Tecw * np.std(Te, axis=0)
        dist_mat      = xp.zeros((len(Te),n_grids))
        
        print("K-means Te clusters ", Te_b)                
        for i in range(self.param.n_grids):
            dist_mat[:,i] = xp.abs(Tew-Tecw[i])
        
        membership = xp.argmin(dist_mat, axis=1)
        grid_idx_to_spatial_pts_map = list()
        for b_idx in range(self.param.n_grids):
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
        self.bte_solver                  = BoltzmannSolver(self.param, ev_max , Te , nr, lm_modes, self.param.n_grids, self.param.collisions)

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
        
        
        active_grid_idx=list()
        for grid_idx in range(n_grids):
            spec_sp     = self.bte_solver._op_spec_sp[grid_idx]
            ev_max_ext  = (spec_sp._basis_p._t[-1] * vth[grid_idx] / self.param.c_gamma)**2
            if  ev_max_ext > 15.76:
                active_grid_idx.append(grid_idx)
        
        self.active_grid_idx  = active_grid_idx #[i for i in range(self.param.n_grids)]
        #self.active_grid_idx  = [i for i in range(self.param.n_grids)]
        self.sub_clusters_run = False
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
    
    async def fetch(self, interface, use_interp:bool):
        gidx_to_pidx            = self.grid_idx_to_spatial_idx_map
        heavy_temp              = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts                = len(heavy_temp)
        self.tps_npts           = tps_npts
        nspecies                = interface.Nspecies()
        electron_temp           = np.array(interface.HostRead(libtps.t2bIndex.ElectronTemperature), copy=False)
        efield                  = np.array(interface.HostRead(libtps.t2bIndex.ElectricField), copy=False).reshape((2, tps_npts))
        species_densities       = np.array(interface.HostRead(libtps.t2bIndex.SpeciesDensities), copy=False).reshape(nspecies, tps_npts)
        
        # np.save("n0.npy", species_densities[TPSINDEX.NEU_IDX])
        # np.save("ne.npy", species_densities[TPSINDEX.ELE_IDX])
        # np.save("ni.npy", species_densities[TPSINDEX.ION_IDX])
        
        # np.save("Te.npy", heavy_temp)
        # np.save("Tg.npy", heavy_temp)
        # np.save("E.npy" , np.sqrt(efield[0]**2 + efield[1]**2))
        # sys.exit(-1)
        
        n_grids            = self.param.n_grids 
        use_gpu            = self.param.use_gpu
        
        Tg                 = heavy_temp
        n0                 = species_densities[TPSINDEX.NEU_IDX]
        ne                 = species_densities[TPSINDEX.ELE_IDX]
        ni                 = species_densities[TPSINDEX.ION_IDX]
    
        Ex                 = efield[0]
        Ey                 = efield[1]
    
        ExbyN              = Ex/n0/self.param.Td_fac
        EybyN              = Ey/n0/self.param.Td_fac
        
        Ex                 = ExbyN * self.param.n0 * self.param.Td_fac
        Ey                 = EybyN * self.param.n0 * self.param.Td_fac
        
        ion_deg            = np.zeros_like(ne) #ne/n0

        m_bte              = np.concatenate((ExbyN.reshape((-1, 1)), EybyN.reshape((-1, 1)), Tg.reshape((-1, 1)), ion_deg.reshape((-1,1)) ), axis=1)
        
        self.sub_cluster_idx_to_pidx = None
        self.sub_cluster_c           = None          
        gidx_to_device_map           = self.gidx_to_device_map
        
        if (use_interp == True):
            n_sub_clusters               = self.param.n_sub_clusters
            self.sub_cluster_idx_to_pidx = [[None for i in range(n_sub_clusters)] for i in range(self.param.n_grids)]
            self.sub_cluster_c           = [None for i in range(self.param.n_grids)]
            
            def normalize(obs):
                std_obs   = np.std(obs, axis=0)
                std_obs[std_obs == 0.0] = 1.0
                return obs/std_obs, std_obs

            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
                def t1():
                    dev_id                       = self.gidx_to_device_map(grid_idx, n_grids)
                    m                            = m_bte[gidx_to_pidx[grid_idx]]
                    mw , mw_std                  = normalize(m)
                    mcw0                         = mw[np.random.choice(mw.shape[0], self.param.n_sub_clusters, replace=False)]
                    mcw                          = scipy.cluster.vq.kmeans(mw, mcw0, iter=1000, thresh=1e-8, check_finite=False)[0]
                    mcw0[0:mcw.shape[0], :]      = mcw[:,:]
                    mcw                          = mcw0
                    
                    mc                           = mcw * mw_std
                    dist_mat                     = np.array([np.linalg.norm(mw - mcw[i], axis=1) for i in range(n_sub_clusters)]).T
                    membership_m                 = np.argmin(dist_mat, axis=1)
                    self.sub_cluster_c[grid_idx] = mc
                    
                    for c_idx in range(n_sub_clusters):
                        self.sub_cluster_idx_to_pidx[grid_idx][c_idx] = np.argwhere(membership_m==c_idx)[:,0]
                        
                        # idx     = self.sub_cluster_idx_to_pidx[grid_idx][c_idx]
                        # abs_err = np.linalg.norm(dist_mat[idx, c_idx] - np.linalg.norm(mw[idx] - mcw[c_idx], axis=1))
                        # print(grid_idx, c_idx, abs_err)
                    
                    # dw_mat = np.zeros(self.param.n_sub_clusters)
                    # print(grid_idx,"\n" , mc)
                    # for c_idx in range(n_sub_clusters):
                    #     idx = self.sub_cluster_idx_to_pidx[grid_idx][c_idx]
                    #     if len(idx>0):
                    #         dw_mat[c_idx] =  np.max(np.linalg.norm(1 - m[idx] / mc[c_idx], axis = 1))
                    
                    # plt.figure(figsize=(8, 8), dpi=300)
                    # plt.semilogy(np.array(range(self.param.n_sub_clusters)), dw_mat)
                    # plt.xlabel(r"cluster id")
                    # plt.ylabel(r"relative error")
                    # plt.grid(visible=True)
                    # plt.savefig("%s_grid_idx_%04d.png"%(self.param.out_fname, grid_idx))
                    # plt.close()
                    
                    
                    n0   = np.ones(mc.shape[0]) * self.param.n0
                    Ex   = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey   = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    
                    Tg   = mc[: , 2]
                    ne   = mc[: , 3] * self.param.n0
                    ni   = mc[: , 3] * self.param.n0
                    EMag = np.sqrt(Ex**2 + Ey**2)
                    
                    if self.param.verbose == 1 :
                        print("Boltzmann solver inputs for v-space grid id %d"%(grid_idx))
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq))
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx])
                        
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(np.min(ExbyN), np.max(ExbyN)))
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(np.min(EybyN), np.max(EybyN)))
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(np.min(Tg)   , np.max(Tg)))
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(np.min(ne)   , np.max(ne)))
                                            
                    if (use_gpu==1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                    
                        
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    
                    return
                
            await ts
            
        else:
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
                def t1():
                    bte_idx           = gidx_to_pidx[grid_idx]
                    dev_id            = self.gidx_to_device_map(grid_idx, n_grids)
                    
                    mc                = m_bte[bte_idx]
                    
                    n0                = np.ones(mc.shape[0]) * self.param.n0
                    Ex                = mc[: , 0] * self.param.n0 * self.param.Td_fac
                    Ey                = mc[: , 1] * self.param.n0 * self.param.Td_fac
                    
                    Tg                = mc[: , 2]
                    ne                = mc[: , 3] * self.param.n0
                    ni                = mc[: , 3] * self.param.n0
                    EMag              = np.sqrt(Ex**2 + Ey**2)
                
                    if self.param.verbose == 1 :
                        print("Boltzmann solver inputs for v-space grid id %d"%(grid_idx))
                        print("Efreq = %.4E [1/s]" %(self.param.Efreq))
                        print("n_pts = %d" % self.grid_idx_to_npts[grid_idx])
                        
                        print("Ex/n0 (min)               = %.12E [Td]         \t Ex/n0(max) = %.12E [Td]    "%(np.min(ExbyN), np.max(ExbyN)))
                        print("Ey/n0 (min)               = %.12E [Td]         \t Ey/n0(max) = %.12E [Td]    "%(np.min(EybyN), np.max(EybyN)))
                        print("Tg    (min)               = %.12E [K]          \t Tg   (max) = %.12E [K]     "%(np.min(Tg)   , np.max(Tg)))
                        print("ne    (min)               = %.12E [1/m^3]      \t ne   (max) = %.12E [1/m^3] "%(np.min(ne)   , np.max(ne)))
            
            
                    if (use_gpu == 1):
                        with cp.cuda.Device(dev_id):
                            n0   = cp.array(n0) 
                            ne   = cp.array(ne)
                            ni   = cp.array(ni)
                            Ex   = cp.array(Ex)
                            Ey   = cp.array(Ey)
                            Tg   = cp.array(Tg)
                            EMag = cp.sqrt(Ex**2 + Ey**2)
                    
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "n0" , n0)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ne" , ne)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "ni" , ni)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "Tg" , Tg)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eRe", Ex)
                    self.bte_solver.set_boltzmann_parameter(grid_idx, "eIm", Ey)
                    self.bte_solver.set_boltzmann_parameter(grid_idx,  "E" , EMag)
                    return
            await ts
        return        

    async def solve_init(self, use_interp:bool):
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
            
            if (use_interp==True):
                n_pts                               = self.param.n_sub_clusters
            else:
                n_pts                               = f_mw.shape[1]
                
            Qmat                                    = self.bte_solver._op_qmat[grid_idx]
            INr                                     = xp.eye(Qmat.shape[1])
            self.bte_solver._op_imat_vx[grid_idx]   = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr)
            
        if(self.param.use_gpu==1):
            self.xp_module = cp
            
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                dev_id = gidx_to_device_map(grid_idx, n_grids)
                @spawn(ts[grid_idx], placement=[gpu(dev_id)], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    f_mw = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    
                    if (use_interp==True):
                        self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", cp.copy(f_mw[: , 0:self.param.n_sub_clusters]))
                    else:
                        self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", cp.copy(f_mw))
                    
            
            await ts
        else:
            self.xp_module = np
            ts = TaskSpace("T")
            for grid_idx in self.active_grid_idx:
                @spawn(ts[grid_idx], placement=[cpu], vcus=0.0)
                def t1():
                    ts_op_setup(grid_idx)
                    f_mw = self.bte_solver.get_boltzmann_parameter(grid_idx, "f_mw")
                    if (use_interp==True):
                        self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", np.copy(f_mw[: , 0:self.param.n_sub_clusters]))
                    else:
                        self.bte_solver.set_boltzmann_parameter(grid_idx, "u0", np.copy(f_mw))
            
            await ts
        
        return
            
    async def solve_step(self, time, delta_t):
        """
        perform a single timestep in 0d-BTE
        """
        rank                    = self.comm.Get_rank()
        npes                    = self.comm.Get_size()
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
    
    async def push(self, interface, use_interp:bool):
        xp                      = self.xp_module
        n_grids                 = self.param.n_grids
        gidx_to_device_map      = self.gidx_to_device_map
        gidx_to_pidx_map        = self.grid_idx_to_spatial_idx_map
        
        heavy_temp  = np.array(interface.HostRead(libtps.t2bIndex.HeavyTemperature), copy=False)
        tps_npts    = len(heavy_temp)
        
        n_reactions = interface.nComponents(libtps.t2bIndex.ReactionRates)
        rates       = np.array(interface.HostWrite(libtps.t2bIndex.ReactionRates), copy=False).reshape((n_reactions, tps_npts))
        
        if (use_interp==True):
            if(n_reactions>0):
                rates[:,:] = 0.0
                ts = TaskSpace("T")
                for grid_idx in self.active_grid_idx:
                    @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                    def t1():
                        qA        = boltzmann.bte_solver._op_diag_dg[grid_idx]
                        u0        = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        
                        h_curr    = xp.dot(qA, u0)
                        h_curr    = boltzmann.bte_solver.normalized_distribution(grid_idx, h_curr)
                        qoi       = boltzmann.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        
                        rr_interp = np.zeros(len(gidx_to_pidx_map[grid_idx]))
                        rr_cpu    = xp.asnumpy(qoi["rates"][TPSINDEX.RR_IDX[0]])
                        
                        for c_idx in range(self.param.n_sub_clusters):
                            rr_interp[self.sub_cluster_idx_to_pidx[grid_idx][c_idx]] = rr_cpu[c_idx] * self.param.N_Avo
                        
                        rates[0][gidx_to_pidx_map[grid_idx]] = rr_interp
                        
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
                        qA       = boltzmann.bte_solver._op_diag_dg[grid_idx]
                        u0       = boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
                        
                        h_curr   = xp.dot(qA, u0)
                        h_curr   = boltzmann.bte_solver.normalized_distribution(grid_idx, h_curr)
                        qoi      = boltzmann.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                        rates[0][gidx_to_pidx_map[grid_idx]] = xp.asnumpy(qoi["rates"][TPSINDEX.RR_IDX[0]]) * self.param.N_Avo
                        
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
        h_curr                  = boltzmann.bte_solver.normalized_distribution(grid_idx, h_curr)
        ff                      = h_curr
        qoi                     = boltzmann.bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)


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
        Tg       = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "Tg"))
        eRe      = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eRe"))
        eIm      = asnumpy(self.bte_solver.get_boltzmann_parameter(grid_idx, "eIm"))
        eMag     = np.sqrt(eRe**2 + eIm**2)
        
        data_csv = np.zeros((ne.shape[0], 8 + len((self.param.collisions))))    

        if export_csv:
            data_csv[: , 0]    = n0
            data_csv[: , 1]    = ne
            data_csv[: , 2]    = ni
            data_csv[: , 3]    = Tg
            data_csv[: , 4]    = eMag
            data_csv[: , 5]    = asnumpy(qoi["energy"])
            data_csv[: , 6]    = asnumpy(qoi["mobility"])
            data_csv[: , 7]    = asnumpy(qoi["diffusion"])
                
            for col_idx, g in enumerate(self.param.collisions):
                data_csv[: , 8 + col_idx]    = asnumpy(qoi["rates"][col_idx])
                
            
            with open("%s_qoi.csv"%(fname), 'w', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter=',')
                # write the header
                header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
                for col_idx, g in enumerate(self.param.collisions):
                    header.append(str(g))
                
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
                    fr = np.abs(ff_r[ii, lm_idx, :])
                    #plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td], $n_e/n_0$ = %.2E "%(Tg[ii], eMag[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]))
                
                #plt.xlabel(r"energy (eV)")
                #plt.ylabel(r"$f_%d$"%(lm[0]))
                plt.grid(visible=True)
                if lm_idx==0:
                    plt.legend(prop={'size': 6})
                    
                plt_idx +=1
            
            plt.savefig("%s_plot.png"%(fname))
            plt.close()
            
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
            
            bte_use_interp = True
            await boltzmann.solve_init(bte_use_interp)
            xp = boltzmann.bte_solver.xp_module

            max_iters = tps.getRequiredInput("cycle-avg-joule-coupled/max-iters")
            iter      = 0
            tt        = 0#interface.currentTime()
            tau       = (1/boltzmann.param.Efreq)
            dt_tps    = interface.timeStep()
            dt_bte    = 1e-2 * tau #boltzmann.param.dt * (dt_tps)
            bte_steps = int(dt_tps/dt_bte)
            n_grids   = boltzmann.param.n_grids
            
            cycle_freq           = 1 #int(xp.ceil(tau/dt_tps))
            terminal_output_freq = -1
            gidx_to_device_map = boltzmann.gidx_to_device_map
            
            tps_sper_cycle = int(xp.ceil(tau/dt_tps))
            bte_sper_cycle = int(xp.ceil(tau/dt_bte))
            bte_max_cycles = 10
            tps_max_cycles = 1000
            
            print("tps steps per cycle : ", tps_sper_cycle, "bte_steps per cycle", bte_sper_cycle)
            tps.solveStep()
            tps.push(interface)
            p_t1 = 0 
            p_t2 = 0
            while (iter<max_iters):
                
                if (iter%cycle_freq==0):
                    interface.saveDataCollection(cycle=(iter//cycle_freq), time=iter)
                
                ########################## BTE solve ##################################################
                await boltzmann.fetch(interface, use_interp=bte_use_interp)
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
                                
                                #boltzmann.io_output_data(grid_idx, u0, False, True, fname)
                                
                                abs_error[grid_idx] = xp.max(xp.abs(bte_v[grid_idx]-u0))
                                rel_error[grid_idx] = abs_error[grid_idx] / xp.max(xp.abs(u0))
                                #print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E"%(bte_idx, tt_bte, abs_error[grid_idx], rel_error[grid_idx]))
                                
                                # if(bte_idx >0):
                                #     print(grid_idx, " u_ptr ", u_avg[grid_idx].data, " v_ptr " , v_avg[grid_idx].data)
                                
                                bte_v[grid_idx] = xp.copy(u0)
                                
                        await ts
                        p_t3 = min_mean_max(p_t2-p_t1, comm)
                        print("[BTE] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E --- runtime = %.4E (s) "%(bte_idx, tt_bte, max(abs_error), max(rel_error), p_t3[2]))
                        
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
                    
                    p_t1 = time()
                    await boltzmann.solve_step(tt_bte, dt_bte)
                    p_t2 = time()
                    
                    if(terminal_output_freq > 0 and bte_idx % terminal_output_freq ==0):
                        p_t3 = min_mean_max(p_t2-p_t1, comm)
                        print("[BTE] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (bte_idx, tt_bte, p_t3[0], p_t3[1], p_t3[2]))
                    
                    ts = TaskSpace("T")
                    for grid_idx in boltzmann.active_grid_idx:
                        @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                        def t1():
                            u_avg[grid_idx] += cycle_f1 * boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1")
                            boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u0", boltzmann.bte_solver.get_boltzmann_parameter(grid_idx, "u1"))
                    await ts
                
                    tt_bte += dt_bte
                
                ts = TaskSpace("T")
                for grid_idx in boltzmann.active_grid_idx:
                    @spawn(ts[grid_idx], placement=[gpu(gidx_to_device_map(grid_idx,n_grids))], vcus=0.0)
                    def t1():
                        boltzmann.bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", u_avg[grid_idx])
                await ts
                await boltzmann.push(interface, use_interp=bte_use_interp)
                
                
                ################### tps solve ######################################
                tps.fetch(interface)
                tps_u  = 0
                tps_v  = 0
                tt_tps = 0
                
                p_t1   = 0
                p_t2   = 0
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
                        
                        p_t3 = min_mean_max(p_t2-p_t1, comm)
                        print("[TPS] step = %04d time = %.4E ||u1 - u0|| = %.4E ||u0 - u1|| / ||u0|| = %.4E -- runtime = %.4E (s)"%(tps_idx, tt_tps, np.max(abs_error), np.max(rel_error), p_t3[2]))
                        if (np.max(abs_error) < boltzmann.param.atol or np.max(rel_error) < max(1e-6,boltzmann.param.rtol)):
                            break
                    
                    if (tps_idx == tps_sper_cycle * tps_max_cycles):
                        break
                    
                    p_t1 = time()
                    tps.solveStep()
                    p_t2 = time()
                    if(terminal_output_freq > 0 and tps_idx % terminal_output_freq ==0):
                        p_t3 = min_mean_max(p_t2-p_t1, comm)
                        print("[TPS] %04d simulation time = %.4E cycle step (min) = %.4E (s) step (mean) = %.4E (s) step (max) = %.4E (s)" % (tps_idx,tt_tps, p_t3[0],p_t3[1],p_t3[2]))
                    tt_tps +=dt_tps
                
                tps.push(interface)
                
                tt += dt_tps * tps_idx
                iter+=1
        

    tps.solveEnd()
    sys.exit (tps.getStatus())