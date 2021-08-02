#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([tps], [1.0], [https://github.com/pecos/tps])
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_HEADERS([tps_config.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE

# Checks for programs.
#AM_PATH_PYTHON([3.6.8])

AX_SUMMARIZE_ENV
#AC_PROG_CXX
AX_PROG_CC_MPI
AX_PROG_CXX_MPI

# C++ 17 standard, just setting by hand
#AX_CXX_COMPILE_STDCXX_17
#CXXFLAGS="-std=c++17 $CXXFLAGS"
CXXFLAGS="-std=c++11 $CXXFLAGS"

# 3rd party libraries
AX_PATH_GRVY([0.32],[yes])
AX_PATH_HDF5([1.8.0],[yes])

#AC_PROG_LIBTOOL
AC_PROG_RANLIB

#-- Slurm job restart support
ENABLE_SLURM=no
AH_TEMPLATE([HAVE_SLURM],[Enable SLURM resource manager support])
AC_ARG_ENABLE(slurm, [AS_HELP_STRING([--enable-slurm Whether to enable optional SLURM support or not (default=check)])],
                     [],[enable_slurm=check])

if test "x$enable_slurm" != "xno" ; then
   AC_CHECK_HEADER([slurm/slurm.h],[found_slurm_header=yes],[found_slurm_header=no])
   if test "x${found_slurm_header}" = "xyes" ; then
      AC_CHECK_LIB([slurm],slurm_get_rem_time,[found_slurm_library=yes],[found_slurm_library=no])
      if test "x${found_slurm_library}" = "xyes" ; then
         AC_DEFINE([HAVE_SLURM])
         ENABLE_SLURM=yes
      else
         ENABLE_SLURM=no
      fi
   fi
   if test "x${enable_slurm}" = "xyes" ; then
       if test "x$ENABLE_SLURM" = "xno" ; then
           AC_MSG_ERROR([slurm requested, but not found locally])
       fi
   fi
fi
AM_CONDITIONAL(SLURM_ENABLED,test x$ENABLE_SLURM = xyes)

#-- GPU/Cuda build setup
AH_TEMPLATE([_GPU_],[Build in GPU support])
AC_MSG_CHECKING([if requesting gpu build])
AC_ARG_ENABLE([gpu-cuda],
         [AS_HELP_STRING([--enable-gpu-cuda],[Enable GPU build, requires associated MFEM with CUDA support.])],
	 [],[enable_gpu_cuda=no])

AC_ARG_VAR(CUDA_ARCH,"Specifies target architecture for CUDA build")

if test x$enable_gpu_cuda = xyes ;then


AC_MSG_RESULT([yes])
AC_DEFINE([_GPU_])

# make sure we were given a desired cuda arch
AC_MSG_CHECKING([if gpu arch was provided])
if test x$CUDA_ARCH = x; then
AC_MSG_RESULT([no])
echo " "
echo "A desired CUDA_ARCH is required. Example settings include:"
echo "CUDA_ARCH=sm_70        # LLNL Lassen"
echo "CUDA_ARCH=sm_75        # Marvin"
echo " "
AC_MSG_ERROR([Please rerun configure with a valid CUDA_ARCH setting.])
else
AC_MSG_RESULT([yes])
fi

AC_LANG_PUSH([C])
AX_CHECK_CUDA
AC_LANG_POP([C])

CXX=$NVCC_PATH/nvcc
CUDA_CXXFLAGS="-x=cu --expt-extended-lambda -arch=$CUDA_ARCH -ccbin mpicxx $CUDA_CXXFLAGS"
CUDA_LDFLAGS="--expt-extended-lambda -arch=$CUDA_ARCH -ccbin mpicxx $CUDA_LDFLAGS"

ENABLE_GPU=yes
ENABLE_CUDA=yes

else
AC_MSG_RESULT([no])
ENABLE_GPU=no
ENABLE_CUDA=no
fi
AM_CONDITIONAL(CUDA_ENABLED,test x$ENABLE_CUDA = xyes)

#--

AC_OUTPUT([Makefile src/Makefile test/Makefile])

# Final summary
AX_SUMMARIZE_CONFIG
