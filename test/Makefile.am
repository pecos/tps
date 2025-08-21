AM_CPPFLAGS = -g -I$(top_srcdir)/src $(MFEM_CXXFLAGS) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
AM_LDFLAGS = $(MPI4PY_LIBS) $(PYTHON_LIBS) $(MFEM_LIBS)

if CUDA_ENABLED
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
AM_LDFLAGS += $(CUDA_LDFLAGS)
endif

if HIP_ENABLED
AM_CPPFLAGS += $(HIP_CXXFLAGS)
endif

if MASA_ENABLED
AM_CPPFLAGS += $(MASA_CXXFLAGS)
AM_LDFLAGS += $(MASA_LIBS)
endif

if GSL_ENABLED
AM_CPPFLAGS += $(GSL_CFLAGS)
AM_LDFLAGS += $(GSL_LIBS)
endif

TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/test/tap-driver.sh

EXTRA_DIST  = tap-driver.sh test_tps_splitcomm.py soln_differ inputs meshes lte-data rad-data \
	          ref_solns/*.h5 ref_solns/collisions/*.h5 ref_solns/transport/*.h5 \
			  ref_solns/reaction/*.h5 ref_solns/noSGM1K/*.h5 ref_solns/smag1K_1/*.h5 \
			  ref_solns/sigma1K_1/*.h5 ref_solns/plate150K/*.h5 ref_solns/plate150K_1step/*.h5 \
              ref_solns/heatedBox/*.h5 ref_solns/spongeBox/*.h5 ref_solns/interpInlet/*.h5 \
              ref_solns/sgsLoMach/*.h5 \
	      ref_solns/lequere-varmu/*.h5 \
	      ref_solns/taylor-couette/*.h5 \
	      ref_solns/pipe/*.h5 \
	      ref_solns/aveLoMach/*.h5 \
              ref_solns/reactBinDiff/*.h5 \
              ref_solns/reactSingleRx/*.h5 \
              ref_solns/reactTable/*.h5 \
              ref_solns/radDecay/*.h5 \
              ref_solns/stabChan/*.h5 \
		      vpath.sh die.sh count_gpus.sh sniff_mpirun.sh \
		      cyl3d.gpu.test cyl3d.mflow.gpu.test wedge.gpu.test \
	          averaging.gpu.test cyl3d.test cyl3d.gpu.python.test cyl3d.mflow.test cyl3d.dtconst.test \
		      die.test averaging.test wedge.test cyl3d.interp.test qms.rings.test qms.axisym.test \
		      sponge_zone.test annulus.test pipe.test valgrind.test valgrind.suppressions \
		      mms.euler.test reaction.test perfect_gas.test collision_integrals.test \
                      diffusion_wall.test \
	 	      inflow_outflow.test mms.ternary_2d.test mms.ternary_2d_wall.test \
	 	      mms.ternary_2d_inout.test cuda-memcheck.test mms.general_wall.test \
		      independent_coupling.test interp_em.test mms.euler_2d.test mms.cns_2d.test \
		      gradient.test coupled-3d.test coupled-3d.splitcomm.test coupled-3d.py-loop.test \
		      tabulated.test lte_mixture.test distance_fcn.test \
              sgsSmag.test sgsSigma.test heatEq.test sponge.test plate.test pipe.mix.test lte2noneq-restart.test \
              coupled-3d.interface.test plasma.axisym.test plasma.axisym.lte1d.test \
	      lomach-flow.test lomach-lequere.test interpInlet.test sgsLoMach.test autoPeriodic.test aveLoMach.test \
              reactFlow-binDiff.test reactFlow-singleRx.test reactFlow-table.test radiativeDecay.test lomach-chan-stab.test \
              reactFlow-nitrogen.test
#	 	      argon_minimal.test argon_minimal.binary.test

TESTS = vpath.sh
XFAIL_TESTS =

if GPU_ENABLED
TESTS += cyl3d.gpu.test \
	 cyl3d.mflow.gpu.test \
	 wedge.gpu.test \
	 averaging.gpu.test \
	 diffusion_wall.test \
	 reaction.test \
	 inflow_outflow.test \
	 pipe.test \
	 sgsSmag.test \
	 sgsSigma.test \
	 heatEq.test \
         plate.test \
	 pipe.mix.test \
	 plasma.axisym.lte1d.test \
         sgsLoMach.test \
	 lomach-flow.test \
	 lomach-lequere.test
#	 argon_minimal.binary.test \

if PYTHON_ENABLED
TESTS += cyl3d.gpu.python.test
endif

else
TESTS += cyl3d.test \
	 cyl3d.mflow.test \
	 cyl3d.dtconst.test \
	 averaging.test \
	 wedge.test \
	 qms.rings.test \
	 qms.axisym.test \
	 sponge_zone.test \
	 annulus.test \
	 pipe.test \
	 perfect_gas.test \
	 collision_integrals.test \
	 diffusion_wall.test \
	 reaction.test \
	 inflow_outflow.test \
	 independent_coupling.test \
	 interp_em.test \
	 gradient.test \
	 coupled-3d.test \
	 coupled-3d.splitcomm.test \
	 tabulated.test \
	 lte_mixture.test \
         sgsSmag.test \
         sgsSigma.test \
         heatEq.test \
	 sponge.test \
         sgsLoMach.test \
         plate.test \
         pipe.mix.test \
	 lte2noneq-restart.test \
	 coupled-3d.interface.test \
	 plasma.axisym.test \
	 plasma.axisym.lte1d.test \
	 lomach-flow.test \
	 lomach-lequere.test \
         interpInlet.test \
         autoPeriodic.test \
         aveLoMach.test \
         reactFlow-binDiff.test \
         reactFlow-singleRx.test \
         reactFlow-table.test \
         radiativeDecay.test \
         lomach-chan-stab.test \
         reactFlow-nitrogen.test
#	 argon_minimal.test \
#	 argon_minimal.binary.test \

if PYTHON_ENABLED
TESTS += cyl3d.python.test \
         cyl3d.python.splitcomm.test \
		 coupled-3d.py-loop.test
endif

endif

if GSLIB_ENABLED
TESTS += cyl3d.interp.test
endif

check_PROGRAMS =

if VALGRIND_ENABLED
check_PROGRAMS += uninit
uninit_SOURCES = badloop.cpp
if !CUDA_ENABLED
TESTS  += valgrind.test
endif
endif

if CUDA_ENABLED
check_PROGRAMS  += badcuda
badcuda_SOURCES  = badcuda.cpp
TESTS           += cuda-memcheck.test
endif

if MASA_ENABLED
TESTS += mms.euler.test \
	 mms.ternary_2d.test \
	 mms.ternary_2d_wall.test \
	 mms.ternary_2d_inout.test
if !GPU_ENABLED
TESTS += mms.general_wall.test \
	 mms.euler_2d.test \
	 mms.cns_2d.test
endif
endif

check_PROGRAMS += tps_splitcomm_test
tps_splitcomm_test_SOURCES  = test_tps_splitcomm.cpp
tps_splitcomm_test_LDADD    = ../src/libtps.la
tps_splitcomm_test_LDADD   += $(HDF5_LIBS)
tps_splitcomm_test_LDADD   += $(GRVY_LIBS)

check_PROGRAMS += tps_interface_test
tps_interface_test_SOURCES  = test_tps_interface.cpp
tps_interface_test_LDADD    = ../src/libtps.la
tps_interface_test_LDADD   += $(HDF5_LIBS)
tps_interface_test_LDADD   += $(GRVY_LIBS)

check_PROGRAMS += split_interface_test
split_interface_test_SOURCES  = test_split_interface.cpp
split_interface_test_LDADD    = ../src/libtps.la
split_interface_test_LDADD   += $(HDF5_LIBS)
split_interface_test_LDADD   += $(GRVY_LIBS)

#check_PROGRAMS += tomboulides_test
#tomboulides_test_SOURCES  = test_tomboulides.cpp
#tomboulides_test_LDADD    = ../src/libtps.la
#tomboulides_test_LDADD   += $(HDF5_LIBS)
#tomboulides_test_LDADD   += $(GRVY_LIBS)


if !GPU_ENABLED

check_PROGRAMS               += perfect_mixture_test
perfect_mixture_test_SOURCES  = test_perfect_mixture.cpp
perfect_mixture_test_LDADD    = ../src/libtps.la
perfect_mixture_test_LDADD   += $(HDF5_LIBS)
perfect_mixture_test_LDADD   += $(GRVY_LIBS)

check_PROGRAMS               += test_speed_of_sound
test_speed_of_sound_SOURCES   = test_speed_of_sound.cpp
test_speed_of_sound_LDADD     = ../src/libtps.la
test_speed_of_sound_LDADD     += $(HDF5_LIBS)
test_speed_of_sound_LDADD     += $(GRVY_LIBS)

check_PROGRAMS               += test_collision
test_collision_SOURCES        = test_collision_integral.cpp
test_collision_LDADD          = ../src/libtps.la
test_collision_LDADD         += $(HDF5_LIBS)
test_collision_LDADD         += $(GRVY_LIBS)

#check_PROGRAMS               += test_argon_minimal
#test_argon_minimal_SOURCES    = test_argon_minimal.cpp
#test_argon_minimal_LDADD      = ../src/libtps.la
#test_argon_minimal_LDADD     += $(HDF5_LIBS)
#test_argon_minimal_LDADD     += $(GRVY_LIBS)

check_PROGRAMS               += test_bdr_flux
test_bdr_flux_SOURCES  	      = test_boundary_flux.cpp
test_bdr_flux_LDADD    	      = ../src/libtps.la
test_bdr_flux_LDADD    	     += $(HDF5_LIBS)
test_bdr_flux_LDADD    	     += $(GRVY_LIBS)

check_PROGRAMS               += test_interp_em
test_interp_em_SOURCES        = test_interp_em.cpp
test_interp_em_LDADD          = ../src/libtps.la
test_interp_em_LDADD         += $(HDF5_LIBS)
test_interp_em_LDADD         += $(GRVY_LIBS)

check_PROGRAMS               += test_grad
test_grad_SOURCES             = test_gradient.cpp
test_grad_LDADD               = ../src/libtps.la
test_grad_LDADD              += $(HDF5_LIBS)
test_grad_LDADD              += $(GRVY_LIBS)

check_PROGRAMS               += test_table
test_table_SOURCES            = test_table.cpp
test_table_LDADD              = ../src/libtps.la
test_table_LDADD             += $(HDF5_LIBS)
test_table_LDADD             += $(GRVY_LIBS)

check_PROGRAMS               += test_lte_mixture
test_lte_mixture_SOURCES      = test_lte_mixture.cpp
test_lte_mixture_LDADD        = ../src/libtps.la
test_lte_mixture_LDADD       += $(HDF5_LIBS)
test_lte_mixture_LDADD       += $(GRVY_LIBS)

check_PROGRAMS               += test_distance_fcn
test_distance_fcn_SOURCES     = test_distance_fcn.cpp
test_distance_fcn_LDADD       = ../src/libtps.la
test_distance_fcn_LDADD      += $(HDF5_LIBS)
test_distance_fcn_LDADD      += $(GRVY_LIBS)

check_PROGRAMS               += standalone_chemistry
standalone_chemistry_SOURCES  = standalone_chemistry.cpp
standalone_chemistry_LDADD    = ../src/libtps.la
standalone_chemistry_LDADD   += $(HDF5_LIBS)
standalone_chemistry_LDADD   += $(GRVY_LIBS)

endif # if !GPU_ENABLED
