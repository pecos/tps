AM_CPPFLAGS  = -g $(MFEM_CXXFLAGS) $(HDF5_CFLAGS) $(GRVY_CFLAGS)

if CUDA_ENABLED
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
endif

if HIP_ENABLED
AM_CPPFLAGS += $(HIP_CXXFLAGS)
endif

if GSLIB_ENABLED
AM_CPPFLAGS += -I$(GSLIB_INC)
endif

if MASA_ENABLED
AM_CPPFLAGS += $(MASA_CXXFLAGS)
endif

TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/test/tap-driver.sh

EXTRA_DIST      = tap-driver.sh soln_differ inputs meshes ref_solns/*.h5 \
		  vpath.sh die.sh cyl3d.gpu.test cyl3d.mflow.gpu.test wedge.gpu.test \
	          averages.gpu.test cyl3d.test cyl3d.mflow.test cyl3d.dtconst.test \
		  die.test averages.test wedge.test cyl3d.interp.test qms.rings.test \
		  sponge_zone.test annulus.test pipe.test valgrind.test valgrind.suppressions \
		  mms.euler.test ping_jacobian.test

TESTS = vpath.sh
XFAIL_TESTS =

if GPU_ENABLED
TESTS += cyl3d.gpu.test \
	 cyl3d.mflow.gpu.test \
	 wedge.gpu.test \
	 averages.gpu.test
else
TESTS += cyl3d.test \
	 cyl3d.mflow.test \
	 cyl3d.dtconst.test \
	 die.test \
	 averages.test \
	 wedge.test \
	 qms.rings.test \
	 sponge_zone.test \
	 annulus.test \
	 pipe.test \
	 ping_jacobian.test
endif

if GSLIB_ENABLED
#TESTS += cyl3d.interp.test
endif

check_PROGRAMS =

if VALGRIND_ENABLED
check_PROGRAMS += uninit
uninit_SOURCES = badloop.cpp

TESTS  += valgrind.test
endif

if MASA_ENABLED
TESTS += mms.euler.test
endif

if GPU_ENABLED
# nothing
else
# extra cpu test programs

# local Jacobian tests
check_PROGRAMS += ping_jacobian_local
ping_jacobian_local_SOURCES = ping_jacobian_local.cpp
ping_jacobian_local_LDADD   = ../src/libtps.a
ping_jacobian_local_LDADD  += $(HDF5_LIBS)
ping_jacobian_local_LDADD  += $(GRVY_LIBS)
ping_jacobian_local_LDADD  += $(MFEM_LIBS)

ping_jacobian_local_LDFLAGS =
if SLURM_ENABLED
ping_jacobian_local_LDFLAGS += -lslurm
endif

if GSLIB_ENABLED
ping_jacobian_local_LDADD   += -lgs
ping_jacobian_local_LDFLAGS += -L$(GSLIB_LIB)
endif

if MASA_ENABLED
ping_jacobian_local_LDFLAGS += $(MASA_LIBS)
endif

# global Jacobian tests
check_PROGRAMS += ping_jacobian_global
ping_jacobian_global_SOURCES = ping_jacobian_global.cpp
ping_jacobian_global_LDADD   = ../src/libtps.a
ping_jacobian_global_LDADD  += $(HDF5_LIBS)
ping_jacobian_global_LDADD  += $(GRVY_LIBS)
ping_jacobian_global_LDADD  += $(MFEM_LIBS)

ping_jacobian_global_LDFLAGS =
if SLURM_ENABLED
ping_jacobian_global_LDFLAGS += -lslurm
endif

if GSLIB_ENABLED
ping_jacobian_global_LDADD   += -lgs
ping_jacobian_global_LDFLAGS += -L$(GSLIB_LIB)
endif

if MASA_ENABLED
ping_jacobian_global_LDFLAGS += $(MASA_LIBS)
endif

endif
