#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([tps], [1.1], [https://github.com/pecos/tps])
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_HEADERS([tps_config.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([subdir-objects])
AM_EXTRA_RECURSIVE_TARGETS([style enforcestyle])

AX_SUMMARIZE_ENV
# MPI
AX_PROG_CC_MPI
AX_PROG_CXX_MPI
# libtool/dynamic library support
AC_PROG_LIBTOOL([disable-static])
# python binary
AM_PATH_PYTHON([3.6.8])
# python interface
AC_ARG_ENABLE([pybind11],
         [AS_HELP_STRING([--enable-pybind11],[enable python interface [default=no]])],
	 [],[enable_pybind11=no])
if test ${enable_pybind11} == "yes" ; then
   AX_PYTHON_DEVEL
   CPPFLAGS="$PYTHON_CPPFLAGS $CPPFLAGS"
   AC_LANG_PUSH([C++])
   AC_CHECK_HEADER([pybind11/pybind11.h],[],[found_pybind11_header=no])
   if test "x${found_pybind11_header}" = "xno" ; then
      AC_MSG_ERROR([pybind11 header not found. Please verify pybind11 is installed locally.])
   fi
   AC_LANG_POP([C++])

   # check for mpi4py
   AX_PATH_MPI4PY([3.1], [no])
   if test x$HAVE_MPI4PY = x1 ; then
      ENABLE_MPI4PY=yes
   fi
fi
AM_CONDITIONAL(PYTHON_ENABLED,test x$enable_pybind11 = xyes)
AM_CONDITIONAL(ENABLED_MPI4PY,test x$ENABLE_MPI4PY = xyes)

# enable c++11 standard
CXXFLAGS="-std=c++11 $CXXFLAGS"

# check for mpi-ext.h
AC_LANG_PUSH([C++])
AH_TEMPLATE([HAVE_MPI_EXT],[Enable MPI extensions declared in mpi-ext.h])
AC_CHECK_HEADER([mpi-ext.h],[found_mpi_ext=yes],[found_mpi_ext=no],[#include <mpi.h>])
if test "x${found_mpi_ext}" = "xyes" ; then
   AC_DEFINE([HAVE_MPI_EXT])
fi
AC_LANG_POP([C++])

#--------------------
# 3rd party libraries
#--------------------
AX_PATH_GRVY([0.36], [yes])
AX_PATH_HDF5([1.8.0],[yes])
AX_PATH_OPENBLAS([no])
ENABLE_MASA=no
AX_PATH_MASA([0.50], [no])
if test x$HAVE_MASA = x1 ; then
   ENABLE_MASA=yes
fi
dnl AC_PROG_RANLIB

# Valgrind?
AX_VALGRIND_CHECK()
AM_CONDITIONAL(VALGRIND_ENABLED,test x$enable_valgrind = xyes)

#----------------------------------------------
# Optionally enable code coverage tool
#----------------------------------------------
AX_CODE_COVERAGE


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




AH_TEMPLATE([_GPU_], [Build in GPU support])
AH_TEMPLATE([_CUDA_],[CUDA backend])
AH_TEMPLATE([_HIP_], [HIP backend])

dnl ------------------------------
dnl -- GPU/Cuda build setup

AC_MSG_CHECKING([if requesting gpu-cuda build])
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

AC_ARG_VAR(USER_CUDA_CXXFLAGS,"Additional C++ compiler flags")

CXX=$NVCC_PATH/nvcc
CUDA_CXXFLAGS="-Xcompiler=-fPIC -x=cu --expt-extended-lambda -arch=$CUDA_ARCH $USER_CUDA_CXXFLAGS -ccbin mpicxx $CUDA_CXXFLAGS"
CUDA_LDFLAGS="--expt-extended-lambda -arch=$CUDA_ARCH -ccbin mpicxx $CUDA_LDFLAGS -lcuda -lcudart"

dnl CUDA does not support -fPIC, disable in libtool in favor of -Xcompiler option above
_LT_TAGVAR(lt_prog_compiler_pic,)=''
_LT_TAGVAR(lt_prog_compiler_pic,CXX)=''
dnl CUDA aldo does not support -Wl linker option, replace with -Xlinker
_LT_TAGVAR(lt_prog_compiler_wl,CXX)='-Xlinker '

ENABLE_GPU=yes
ENABLE_CUDA=yes
AC_DEFINE([_CUDA_])

else
AC_MSG_RESULT([no])
ENABLE_GPU=no
ENABLE_CUDA=no
fi
AM_CONDITIONAL(CUDA_ENABLED,test x$ENABLE_CUDA = xyes)
AM_CONDITIONAL(GPU_ENABLED,test x$ENABLE_CUDA = xyes)

dnl ------------------------------
dnl Check for GPU/HIP build setup
ENABLE_HIP=no
if test x$ENABLE_GPU != xyes; then
   AX_HIP_CHECK()
   AM_CONDITIONAL(GPU_ENABLED,test x$ENABLE_HIP = xyes)

   if test x$ENABLE_HIP = xyes; then
       dnl Absolutely insist that   libtool has CC=hipcc for tag=CXX
       _LT_TAGVAR(compiler,CXX)='hipcc'
   fi
fi
AM_CONDITIONAL(HIP_ENABLED,test x$ENABLE_HIP = xyes)


dnl ------------------------------
dnl Check for GPU build without hip or cuda
AC_ARG_ENABLE([gpu-cpu],
         [AS_HELP_STRING([--enable-gpu-cpu],[Enable GPU code path on CPU (WARNING: Experimental option).])],
	 [],[enable_gpu_cpu=no])

if test x$enable_gpu_cpu = xyes; then
   AC_MSG_NOTICE([Configuring for GPU code path build on CPU system])
   AC_MSG_NOTICE([WARNING: This is an experimental feature.])
   AC_DEFINE([_GPU_])
   AM_CONDITIONAL(GPU_ENABLED,test x$enable_gpu_cpu = xyes)
fi



AX_PATH_MFEM([4.4],  [yes])

#-- GSLIB support in MFEM?
AC_MSG_CHECKING([if GSLIB is available in mfem])

ENABLE_GSLIB=no

AC_LANG_PUSH([C++])
cxxflags_save=$CXXFLAGS
CXXFLAGS="$CXXFLAGS $MFEM_CXXFLAGS"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <mfem/config/config.hpp>
#ifndef MFEM_USE_GSLIB
# error macro not defined
#endif
]])], [found_gslib_support="yes"], [found_gslib_support="no"])
CXXFLAGS=$cxxflags_save
AC_LANG_POP([C++])

AH_TEMPLATE([HAVE_GSLIB],[Enable GSLIB support])

if test "x$found_gslib_support" = "xyes" ; then
   AC_DEFINE([HAVE_GSLIB])
   ENABLE_GSLIB=yes
fi

if test "x$ENABLE_GSLIB" = "xno" ; then
   AC_MSG_RESULT([no])
else
   AC_MSG_RESULT([yes])
fi

AM_CONDITIONAL(GSLIB_ENABLED,test x$ENABLE_GSLIB = xyes)


# -------------------------------------------------------------
# Doxygen - look for doxygen (a documentation tool)
# -------------------------------------------------------------
AC_CONFIG_FILES(doc/Doxyfile)
AC_PATH_PROG(DOXYGEN, doxygen)
AC_SUBST(DOXYGEN)
# -------------------------------------------------------------


# -------------------------------------------------------------
# GSL - look for Gnu Scientific Library
# -------------------------------------------------------------
AH_TEMPLATE([HAVE_GSL],[Enable Gnu Scientific Library support])
ENABLE_GSL=no
AX_PATH_GSL(2.5, [ENABLE_GSL=yes], )
if test x$ENABLE_GSL = xyes; then
   AC_DEFINE([HAVE_GSL])
fi
AM_CONDITIONAL(GSL_ENABLED,test x$ENABLE_GSL = xyes)


# -------------------------------------------------------------
# build deprecated capabilities if requested
# -------------------------------------------------------------
AH_TEMPLATE([_BUILD_DEPRECATED_], [Build deprecated functionality])
AC_ARG_ENABLE([build-deprecated],
         [AS_HELP_STRING([--enable-build-deprecated],[Enable support for deprecated functionality.])],
	 [AC_DEFINE([_BUILD_DEPRECATED_])],[])


dnl --

AC_OUTPUT([Makefile src/Makefile test/Makefile utils/Makefile])

dnl create soft link to local bats binary in tests/
AC_MSG_NOTICE([Bootstrapping local bats test configuration])
if test -d test ; then
  if test ! -e test/bats ; then
     if test ! -L test/bats; then
        ln -s $srcdir/../utils/bats/bin/bats test/bats
     fi
  fi
fi

dnl Final summary
AX_SUMMARIZE_CONFIG

