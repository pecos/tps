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

tps_LDFLAGS=

# C++ main solver binary
bin_PROGRAMS         = tps
tps_SOURCES          = main.cpp
tps_LDADD            = libtps.la
tps_LDFLAGS         += $(PYTHON_LIBS) $(MFEM_LIBS)

# C++ solver library sources
cxx_sources          = averaging_and_rms.cpp faceGradientIntegration.cpp M2ulPhyS.cpp rhs_operator.cpp wallBC.cpp \
		       BCintegrator.cpp face_integrator.cpp riemann_solver.cpp BoundaryCondition.cpp fluxes.cpp \
                       masa_handler.cpp run_configuration.cpp domain_integrator.cpp forcing_terms.cpp mpi_groups.cpp \
	               sbp_integrators.cpp equation_of_state.cpp transport_properties.cpp inletBC.cpp outletBC.cpp utils.cpp io.cpp \
		       dgNonlinearForm.cpp  dgNonlinearForm.hpp gradients.cpp gradNonLinearForm.cpp \
		       quasimagnetostatic.cpp tps.cpp chemistry.cpp reaction.cpp collision_integrals.cpp argon_transport.cpp source_term.cpp

mfem_extra_sources   = ../utils/mfem_extras/pfem_extras.cpp

headers              = averaging_and_rms.hpp equation_of_state.hpp transport_properties.hpp forcing_terms.hpp mpi_groups.hpp \
		       run_configuration.hpp BCintegrator.hpp faceGradientIntegration.hpp inletBC.hpp outletBC.hpp \
	               sbp_integrators.hpp BoundaryCondition.hpp face_integrator.hpp M2ulPhyS.hpp rhs_operator.hpp \
	               wallBC.hpp domain_integrator.hpp fluxes.hpp masa_handler.hpp riemann_solver.hpp \
		       dgNonlinearForm.hpp gradNonLinearForm.hpp dataStructures.hpp utils.hpp io.hpp em_options.hpp \
		       quasimagnetostatic.hpp gradients.hpp logger.hpp solver.hpp tps.hpp chemistry.hpp reaction.hpp \
					 collision_integrals.hpp argon_transport.hpp source_term.hpp

mfem_extra_headers   = ../utils/mfem_extras/pfem_extras.hpp

## noinst_LIBRARIES     = libtps.a
## libtps_a_SOURCES     = $(cxx_sources) $(mfem_extra_sources) $(headers) $(mfem_extra_headers)

lib_LTLIBRARIES      = libtps.la
libtps_la_SOURCES    = $(cxx_sources) $(mfem_extra_sources) $(headers) $(mfem_extra_headers)
libtps_la_LIBADD     = $(MFEM_LIBS)
libtps_la_LIBADD    += $(HDF5_LIBS)
libtps_la_LIBADD    += $(GRVY_LIBS)

if SLURM_ENABLED
libtps_la_LIBADD    += -lslurm
endif

if CUDA_ENABLED
tps_LDFLAGS         += $(CUDA_LDFLAGS)
libtps_la_LIBADD    += $(CUDA_LDFLAGS) -lcuda -lcusparse
endif

if HIP_ENABLED
libtps_la_LIBADD    += $(HIP_LDFLAGS)
endif

if GSLIB_ENABLED
libtps_la_LIBADD    += -L$(GSLIB_LIB) -lgs
endif

if MASA_ENABLED
libtps_la_LIBADD += $(MASA_LIBS)
endif


# Python wrapper for C++ solver library
### pyexec_LTLIBRARIES   = pytps.la
### pytps_la_SOURCES     = pytps.cpp
### pytps_la_LIBADD      = -L$(BOOST_LIB) -lboost_python36 -lpython3.6m libtps.la
### pytps_la_LDFLAGS     = -avoid-version -module
### #pytps_la_DEPENDENCIES  = libtps.la
### AM_CPPFLAGS         += -I/usr/include/python3.6m

#---------------------------------
# Linting/style target(s)
#---------------------------------

enforcestyle:
	@echo ""
	@echo "------------------------------------------------------"
	@echo "Applying source code style updates using clang-format"
	@echo "------------------------------------------------------"
	@echo ""
#	@$(top_srcdir)/utils/cpplint/cpplint.py --output=sed $(headers) $(cxx_sources) main.cpp | bash
#	astyle --project $(headers) $(cxx_sources) main.cpp
#	clang-format -i $(headers) $(cxx_sources) main.cpp
	@cd $(top_srcdir)/src && clang-format -i $(headers) $(cxx_sources) main.cpp




style:
	@echo ""
	@echo "-------------------------------------------"
	@echo "Checking source code style using cpplint.py"
	@echo "-------------------------------------------"
	@echo ""
#	@$(top_srcdir)/utils/cpplint/cpplint.py $(headers) $(cxx_sources) main.cpp
	@cd $(top_srcdir)/src && $(abs_top_srcdir)/utils/cpplint/cpplint.py $(headers) $(cxx_sources) main.cpp

#---------------------------------
# Embedded license header support
#---------------------------------
license:
	@echo 'Running license update tool for src files'
	@$(top_srcdir)/utils/update_license.pl -S=$(top_srcdir)/src $(top_srcdir)/LICENSE \
		 $(headers) $(cxx_sources) main.cpp
