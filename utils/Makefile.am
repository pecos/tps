AM_CPPFLAGS = -g -I$(top_srcdir)/src $(MFEM_CXXFLAGS) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
EXTRA_DIST = bats cpplint

if CUDA_ENABLED
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
endif

if HIP_ENABLED
AM_CPPFLAGS += $(HIP_CXXFLAGS)
endif

# Possible there is nothing to do
bin_PROGRAMS =

if GSLIB_ENABLED
# TODO: interpolation utility is temporarily disabled.  It doesn't
# build b/c of changes in M2ulPhyS than haven't been propagated.
#bin_PROGRAMS   += interp
#interp_SOURCES  = pfield_interpolate.cpp
#interp_LDADD    = ../src/libtps.a
#interp_LDADD   += $(HDF5_LIBS)
#interp_LDADD   += $(GRVY_LIBS)
#interp_LDADD   += $(MFEM_LIBS)
endif
