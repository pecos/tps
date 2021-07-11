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

#AC_PROG_CXX
AX_PROG_CXX_MPI

# C++ 17 standard, just setting by hand
#AX_CXX_COMPILE_STDCXX_17
#CXXFLAGS="-std=c++17 $CXXFLAGS"
CXXFLAGS="-std=c++11 $CXXFLAGS"

#AC_PROG_LIBTOOL
AC_PROG_RANLIB

#-- GPU/Cuda build setup
AH_TEMPLATE([_GPU_],[Build in GPU support])
AC_MSG_CHECKING([if requesting gpu build])
AC_ARG_ENABLE([gpu-cuda],
         [AS_HELP_STRING([--enable-gpu-cuda],[Enable GPU build, requires associated MFEM with CUDA support.])],
	 [],[enable_gpu_cuda=no])

if test x$enable_gpu_cuda = xyes ;then

AC_MSG_RESULT([yes])
AC_DEFINE([_GPU_])

AC_LANG_PUSH([C])
AX_CHECK_CUDA
AC_LANG_POP([C])

CXX=$NVCC_PATH/nvcc
#CUDA_CXXFLAGS="-x=cu --expt-extended-lambda -arch=sm_60 -ccbin mpicxx $CUDA_CXXFLAGS"
#CUDA_LDFLAGS="--expt-extended-lambda -arch=sm_60 -ccbin mpicxx $CUDA_LDFLAGS"

CUDA_CXXFLAGS="-x=cu --expt-extended-lambda -arch=sm_70 -ccbin mpicxx $CUDA_CXXFLAGS"
CUDA_LDFLAGS="--expt-extended-lambda -arch=sm_70 -ccbin mpicxx $CUDA_LDFLAGS"

ENABLE_GPU=yes
ENABLE_CUDA=yes

else
AC_MSG_RESULT([no])
ENABLE_GPU=no
ENABLE_CUDA=no
fi
AM_CONDITIONAL(CUDA_ENABLED,test x$ENABLE_CUDA = xyes)

#--

# 3rd party libraries
AX_PATH_GRVY([0.32],[yes])
AX_PATH_HDF5([1.8.0],[yes])


AC_OUTPUT([Makefile src/Makefile test/Makefile])
#AC_OUTPUT([Makefile])

# Final summary
AX_SUMMARIZE_ENV
AX_SUMMARIZE_CONFIG
