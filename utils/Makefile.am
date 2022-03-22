AM_CPPFLAGS          = -g -I../src -I$(MFEM_INC) -I$(MFEM_INC)/mfem
AM_CPPFLAGS         += -I$(HYPRE_INC) $(HDF5_CFLAGS) $(GRVY_CFLAGS)
# AM_CPPFLAGS         += -I$(PETSC_INC)
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
beam_mesh_LDADD    = -lmfem

beam_mesh_LDFLAGS  = -L$(MFEM_LIB)
#beam_mesh_LDFLAGS += -L$(PETSC_LIB) -lpetsc
beam_mesh_LDFLAGS += $(HDF5_LIBS)
beam_mesh_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
beam_mesh_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
beam_mesh_LDFLAGS += -L$(METIS_LIB) -lmetis
beam_mesh_LDFLAGS += $(GRVY_LIBS)
beam_mesh_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
beam_mesh_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
beam_mesh_LDFLAGS += -L$(GSLIB_LIB) -lgs
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
if MASA_ENABLED
interp_LDFLAGS += $(MASA_LIBS)
endif
endif

bin_PROGRAMS += binaryic
binaryic_SOURCES  = binary_mixture_ic.cpp
binaryic_LDADD    = ../src/libtps.a -lmfem

binaryic_LDFLAGS  = -L$(MFEM_LIB)
binaryic_LDFLAGS += $(HDF5_LIBS)
binaryic_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
binaryic_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
binaryic_LDFLAGS += -L$(METIS_LIB) -lmetis
binaryic_LDFLAGS += $(GRVY_LIBS)
binaryic_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
binaryic_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
binaryic_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
binaryic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS += tanhic
tanhic_SOURCES  = tanh_ic.cpp
tanhic_LDADD    = ../src/libtps.a -lmfem

tanhic_LDFLAGS  = -L$(MFEM_LIB)
tanhic_LDFLAGS += $(HDF5_LIBS)
tanhic_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
tanhic_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
tanhic_LDFLAGS += -L$(METIS_LIB) -lmetis
tanhic_LDFLAGS += $(GRVY_LIBS)
tanhic_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
tanhic_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
tanhic_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
tanhic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS += sineic
sineic_SOURCES  = sine_ic.cpp
sineic_LDADD    = ../src/libtps.a -lmfem

sineic_LDFLAGS  = -L$(MFEM_LIB)
sineic_LDFLAGS += $(HDF5_LIBS)
sineic_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
sineic_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
sineic_LDFLAGS += -L$(METIS_LIB) -lmetis
sineic_LDFLAGS += $(GRVY_LIBS)
sineic_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
sineic_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
sineic_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
sineic_LDFLAGS += $(MASA_LIBS)
endif

bin_PROGRAMS += l2diff
l2diff_SOURCES  = L2_diff.cpp
l2diff_LDADD    = ../src/libtps.a -lmfem

l2diff_LDFLAGS  = -L$(MFEM_LIB)
l2diff_LDFLAGS += $(HDF5_LIBS)
l2diff_LDFLAGS += -L$(HYPRE_LIB) -lHYPRE
l2diff_LDFLAGS += -L$(OPENBLAS_LIB) -lopenblas
l2diff_LDFLAGS += -L$(METIS_LIB) -lmetis
l2diff_LDFLAGS += $(GRVY_LIBS)
l2diff_LDFLAGS += -L$(SUPERLU_DIST_LIB) -lsuperlu_dist
l2diff_LDFLAGS += -L$(NETCDF_LIB) -lnetcdf
l2diff_LDFLAGS += -L$(GSLIB_LIB) -lgs
if MASA_ENABLED
l2diff_LDFLAGS += $(MASA_LIBS)
endif

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
