#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([tps], [1.0], [https://github.com/pecos/tps])
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_HEADERS([config.h])
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



AC_OUTPUT([Makefile src/Makefile test/Makefile])

# Final summary
AX_SUMMARIZE_ENV
AX_SUMMARIZE_CONFIG
