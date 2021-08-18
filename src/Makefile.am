AM_CPPFLAGS          = -g -I$(MFEM_INC) -I$(MFEM_INC)/mfem
AM_CPPFLAGS         += -I$(HYPRE_INC) $(HDF5_CFLAGS) $(GRVY_CFLAGS)

if CUDA_ENABLED
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
endif

if HIP_ENABLED
AM_CPPFLAGS += $(HIP_CXXFLAGS)
endif

# C++ main solver binary
bin_PROGRAMS         = tps
tps_SOURCES          = main.cpp
tps_LDADD            = libtps.a -lmfem

tps_LDFLAGS          = -L$(MFEM_LIB)
tps_LDFLAGS         += $(HDF5_LIBS)
tps_LDFLAGS         += -L$(HYPRE_LIB) -lHYPRE
tps_LDFLAGS         += -L$(OPENBLAS_LIB) -lopenblas
tps_LDFLAGS         += -L$(METIS_LIB) -lmetis
tps_LDFLAGS         += $(GRVY_LIBS)
# hypre depends are complicated
#tps_LDFLAGS         += -L$(PETSC_LIB) -lpetsc
tps_LDFLAGS         += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
tps_LDFLAGS         += -L$(NETCDF_LIB) -lnetcdf

if SLURM_ENABLED
tps_LDFLAGS         += -lslurm
endif

if CUDA_ENABLED
tps_LDFLAGS         += $(CUDA_LDFLAGS) -lcuda -lcusparse
endif

if HIP_ENABLED
tps_LDFLAGS         += $(HIP_LDFLAGS)
endif


# C++ solver library sources
cxx_sources          = averaging_and_rms.cpp faceGradientIntegration.cpp M2ulPhyS.cpp rhs_operator.cpp wallBC.cpp \
		       BCintegrator.cpp face_integrator.cpp riemann_solver.cpp BoundaryCondition.cpp fluxes.cpp \
                       masa_handler.cpp run_configuration.cpp domain_integrator.cpp forcing_terms.cpp mpi_groups.cpp \
	               sbp_integrators.cpp equation_of_state.cpp inletBC.cpp outletBC.cpp utils.cpp io.cpp \
		       dgNonlinearForm.cpp  dgNonlinearForm.hpp gradients.cpp gradNonLinearForm.cpp

headers              = averaging_and_rms.hpp equation_of_state.hpp forcing_terms.hpp mpi_groups.hpp \
		       run_configuration.hpp BCintegrator.hpp faceGradientIntegration.hpp inletBC.hpp outletBC.hpp \
	               sbp_integrators.hpp BoundaryCondition.hpp face_integrator.hpp M2ulPhyS.hpp rhs_operator.hpp \
	               wallBC.hpp domain_integrator.hpp fluxes.hpp masa_handler.hpp riemann_solver.hpp \
		       dgNonlinearForm.hpp gradNonLinearForm.hpp dataStructures.hpp io.hpp

noinst_LIBRARIES     = libtps.a
libtps_a_SOURCES     = $(cxx_sources) $(headers)

# disabling dynamic library, did not cooperate with nvcc
#tps_LDADD            = -L$(top_builddir)/src -ltps
#tps_DEPENDENCIES     = libtps.la
#tps_DEPENDENCIES     = libtps.a

## lib_LTLIBRARIES      = libtps.la
## libtps_la_SOURCES    = $(cxx_sources) $(headers)
## libtps_la_LDFLAGS    = -avoid-version
## libtps_la_LDFLAGS   += -L/usr/local/lib -L$(MFEM_LIB) -lmfem
## libtps_la_LDFLAGS   += -L$(HYPRE_LIB) -lHYPRE -L$(PETSC_LIB) -lpetsc -L$(SUPERLU_DIST_LIB) -lsuperlu_dist #-L$(HDF5_LIB) -lhdf5
## libtps_la_LDFLAGS   += -L$(NETCDF_LIB) -lnetcdf
## libtps_la_LDFLAGS   += -lgrvy

# Python wrapper for C++ solver library
### pyexec_LTLIBRARIES   = pytps.la
### pytps_la_SOURCES     = pytps.cpp 
### pytps_la_LIBADD      = -L$(BOOST_LIB) -lboost_python36 -lpython3.6m libtps.la
### pytps_la_LDFLAGS     = -avoid-version -module
### #pytps_la_DEPENDENCIES  = libtps.la
### AM_CPPFLAGS         += -I/usr/include/python3.6m
