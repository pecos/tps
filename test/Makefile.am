AM_CPPFLAGS          = -g -I../src -I$(MFEM_INC) -I$(MFEM_INC)/mfem
AM_CPPFLAGS         += -I$(HYPRE_INC) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
AM_CPPFLAGS         += -I$(PETSC_INC)
#EXTRA_DIST           = bats cpplint

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
		  mms.euler.test reaction.test

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
	 perfect_gas.test \
	 collision_integrals.test \
	 argon_minimal.test \
	 argon_minimal.binary.test \
	 diffusion_wall.test \
	 reaction.test \
	 inflow_outflow.test
endif

if GSLIB_ENABLED
TESTS += cyl3d.interp.test
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

check_PROGRAMS += perfect_mixture_test
perfect_mixture_test_SOURCES  = test_perfect_mixture.cpp
perfect_mixture_test_LDADD    = ../src/libtps.a -lmfem

perfect_mixture_test_LDFLAGS  = -L$(MFEM_LIB)
perfect_mixture_test_LDFLAGS += $(HDF5_LIBS)
perfect_mixture_test_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
perfect_mixture_test_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
perfect_mixture_test_LDFLAGS += -L$(METIS_LIB) -lmetis
perfect_mixture_test_LDFLAGS += $(GRVY_LIBS)
perfect_mixture_test_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
perfect_mixture_test_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
perfect_mixture_test_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
perfect_mixture_test_LDFLAGS += $(MASA_LIBS)
endif

check_PROGRAMS += test_speed_of_sound
test_speed_of_sound_SOURCES  = test_speed_of_sound.cpp
test_speed_of_sound_LDADD    = ../src/libtps.a -lmfem

test_speed_of_sound_LDFLAGS  = -L$(MFEM_LIB)
test_speed_of_sound_LDFLAGS += $(HDF5_LIBS)
test_speed_of_sound_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
test_speed_of_sound_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
test_speed_of_sound_LDFLAGS += -L$(METIS_LIB) -lmetis
test_speed_of_sound_LDFLAGS += $(GRVY_LIBS)
test_speed_of_sound_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
test_speed_of_sound_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
test_speed_of_sound_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
test_speed_of_sound_LDFLAGS += $(MASA_LIBS)
endif

check_PROGRAMS += test_collision
test_collision_SOURCES  = test_collision_integral.cpp
test_collision_LDADD    = ../src/libtps.a -lmfem

test_collision_LDFLAGS  = -L$(MFEM_LIB)
test_collision_LDFLAGS += $(HDF5_LIBS)
test_collision_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
test_collision_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
test_collision_LDFLAGS += -L$(METIS_LIB) -lmetis
test_collision_LDFLAGS += $(GRVY_LIBS)
test_collision_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
test_collision_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
test_collision_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
test_collision_LDFLAGS += $(MASA_LIBS)
endif

check_PROGRAMS += test_argon_minimal
test_argon_minimal_SOURCES  = test_argon_minimal.cpp
test_argon_minimal_LDADD    = ../src/libtps.a -lmfem

test_argon_minimal_LDFLAGS  = -L$(MFEM_LIB)
test_argon_minimal_LDFLAGS += $(HDF5_LIBS)
test_argon_minimal_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
test_argon_minimal_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
test_argon_minimal_LDFLAGS += -L$(METIS_LIB) -lmetis
test_argon_minimal_LDFLAGS += $(GRVY_LIBS)
test_argon_minimal_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
test_argon_minimal_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
test_argon_minimal_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
test_argon_minimal_LDFLAGS += $(MASA_LIBS)
endif

