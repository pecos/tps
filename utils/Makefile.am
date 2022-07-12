AM_CPPFLAGS          = -g -I$(top_srcdir)/src $(MFEM_CXXFLAGS) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
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

if MASA_ENABLED
AM_CPPFLAGS += $(MASA_CXXFLAGS)
endif

# Possible there is nothing to do
bin_PROGRAMS =

bin_PROGRAMS += beam_mesh
beam_mesh_SOURCES  = beam_mesh.cpp
beam_mesh_LDADD    =
beam_mesh_LDADD    += $(HDF5_LIBS)
beam_mesh_LDADD    += $(GRVY_LIBS)
beam_mesh_LDFLAGS   = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
beam_mesh_LDFLAGS += $(MASA_LIBS)
endif

#bin_PROGRAMS += simmesh
#simmesh_SOURCES  = simple_periodic_mesh.cpp
#simmesh_LDADD    = ../src/libtps.a -lmfem
#
#simmesh_LDFLAGS  = -L$(MFEM_LIB)
#simmesh_LDFLAGS += $(HDF5_LIBS)
#simmesh_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
#simmesh_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
#simmesh_LDFLAGS += -L$(METIS_LIB) -lmetis
#simmesh_LDFLAGS += $(GRVY_LIBS)
#simmesh_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
#simmesh_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
#simmesh_LDFLAGS += -L$(GSLIB_LIB) -lgs

if GSLIB_ENABLED
#bin_PROGRAMS   += interp
#interp_SOURCES  = pfield_interpolate.cpp
#interp_LDADD    = ../src/libtps.la
#interp_LDADD    += $(HDF5_LIBS)
#interp_LDADD    += $(GRVY_LIBS)
#interp_LDFLAGS  = $(PYTHON_LIBS) $(MFEM_LIBS)
#if MASA_ENABLED
#interp_LDFLAGS += $(MASA_LIBS)
#endif
endif

bin_PROGRAMS += binaryic
binaryic_SOURCES  = binary_mixture_ic.cpp
binaryic_LDADD    = ../src/libtps.la
binaryic_LDADD   += $(HDF5_LIBS)
binaryic_LDADD   += $(GRVY_LIBS)
binaryic_LDFLAGS  = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
binaryic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS     += tanhic
tanhic_SOURCES    = tanh_ic.cpp
tanhic_LDADD      = ../src/libtps.la -lmfem
tanhic_LDADD     += $(HDF5_LIBS)
tanhic_LDADD     += $(GRVY_LIBS)
tanhic_LDFLAGS    = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
tanhic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS    += sineic
sineic_SOURCES   = sine_ic.cpp
sineic_LDADD     = ../src/libtps.la -lmfem
sineic_LDADD    += $(HDF5_LIBS)
sineic_LDADD    += $(GRVY_LIBS)
sineic_LDFLAGS   = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
sineic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS     += l2diff
l2diff_SOURCES    = L2_diff.cpp
l2diff_LDADD      = ../src/libtps.la -lmfem
l2diff_LDADD     += $(HDF5_LIBS)
l2diff_LDADD     += $(GRVY_LIBS)
l2diff_LDFLAGS    = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
l2diff_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS        += compute_rhs
compute_rhs_SOURCES  = compute_rhs.cpp
compute_rhs_LDADD    = ../src/libtps.la -lmfem
compute_rhs_LDADD   += $(HDF5_LIBS)
compute_rhs_LDADD   += $(GRVY_LIBS)
compute_rhs_LDFLAGS  = $(PYTHON_LIBS) $(MFEM_LIBS)

if MASA_ENABLED
compute_rhs_LDFLAGS += $(MASA_LIBS)
endif

if CUDA_ENABLED
compute_rhs_LDFLAGS += $(CUDA_LDFLAGS)
endif


if !GPU_ENABLED

endif # if !GPU_ENABLED

#bin_PROGRAMS += perfect_mixture_test
#perfect_mixture_test_SOURCES  = test_perfect_mixture.cpp
#perfect_mixture_test_LDADD    = ../src/libtps.a -lmfem
#
#perfect_mixture_test_LDFLAGS  = -L$(MFEM_LIB)
#perfect_mixture_test_LDFLAGS += $(HDF5_LIBS)
#perfect_mixture_test_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
#perfect_mixture_test_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
#perfect_mixture_test_LDFLAGS += -L$(METIS_LIB) -lmetis
#perfect_mixture_test_LDFLAGS += $(GRVY_LIBS)
#perfect_mixture_test_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
#perfect_mixture_test_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
#perfect_mixture_test_LDFLAGS += -L$(GSLIB_LIB) -lgs
#
#bin_PROGRAMS += test_speed_of_sound
#test_speed_of_sound_SOURCES  = test_speed_of_sound.cpp
#test_speed_of_sound_LDADD    = ../src/libtps.a -lmfem
#
#test_speed_of_sound_LDFLAGS  = -L$(MFEM_LIB)
#test_speed_of_sound_LDFLAGS += $(HDF5_LIBS)
#test_speed_of_sound_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
#test_speed_of_sound_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
#test_speed_of_sound_LDFLAGS += -L$(METIS_LIB) -lmetis
#test_speed_of_sound_LDFLAGS += $(GRVY_LIBS)
#test_speed_of_sound_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
#test_speed_of_sound_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
#test_speed_of_sound_LDFLAGS += -L$(GSLIB_LIB) -lgs
