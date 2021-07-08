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
AX_CXX_COMPILE_STDCXX_17
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

#-- GPU build setup
AH_TEMPLATE([_GPU_],[Build in GPU support])
AC_MSG_CHECKING([if requesting gpu build])
AC_ARG_ENABLE([gpu],
         [AS_HELP_STRING([--enable-gpu],[Enable GPU build, requires associated MFEM with gpu support.])],
	 [],[enable_gpu=no])
if test x$enable_gpu = xyes ;then
    AC_MSG_RESULT([yes])
    AC_DEFINE([_GPU_])
    ENABLE_GPU=yes
else
    AC_MSG_RESULT([no])
    ENABLE_GPU=no
fi
#--


AC_OUTPUT([Makefile src/Makefile test/Makefile])

# Final summary
AX_SUMMARIZE_ENV
AX_SUMMARIZE_CONFIG
