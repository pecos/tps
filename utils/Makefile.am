AM_CPPFLAGS          = -g -I../src -I$(MFEM_INC) -I$(MFEM_INC)/mfem
AM_CPPFLAGS         += -I$(HYPRE_INC) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
EXTRA_DIST           = bats cpplint

if CUDA_ENABLED
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
endif

if HIP_ENABLED
AM_CPPFLAGS += $(HIP_CXXFLAGS)
endif

if GSLIB_ENABLED
AM_CPPFLAGS += -I$(GSLIB_INC)
endif

# Possible there is nothing to do
bin_PROGRAMS =

if GSLIB_ENABLED
bin_PROGRAMS   += interp
interp_SOURCES  = pfield_interpolate.cpp
interp_LDADD    = ../src/libtps.a -lmfem

interp_LDFLAGS  = -L$(MFEM_LIB)
interp_LDFLAGS += $(HDF5_LIBS)
interp_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
interp_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
interp_LDFLAGS += -L$(METIS_LIB) -lmetis
interp_LDFLAGS += $(GRVY_LIBS)
interp_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
interp_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
interp_LDFLAGS += -L$(GSLIB_LIB) -lgs
endif
