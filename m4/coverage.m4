# SYNOPSIS
#
#   Add code coverage support with gcov/lcov.
#
#   AX_CODE_COVERAGE()
#
# DESCRIPTION
#
#   Provides a --enable-coverage option which checks for available
#   gcov/lcov binaries and provides ENABLE_CODE_COVERAGE conditional.
#
# LAST MODIFICATION
#
#   $Id$
#
# COPYLEFT
#
#   Copyright (c) 2010 Karl W. Schulz <karl@ices.utexas.edu>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.

AC_DEFUN([AX_CODE_COVERAGE],
[

AC_ARG_ENABLE(coverage, AC_HELP_STRING([--enable-coverage],[configure code coverage analysis tools]))

HAVE_GCOV_TOOLS=0

if test "x$enable_coverage" = "xyes"; then

   # ----------------------------
   # Check for gcov/lcov binaries
   # ----------------------------
   
   AC_CHECK_PROG(have_gcov,gcov, yes, no)

   if test "x$have_gcov" = "xno"; then
      AC_MSG_ERROR([

      gcov coverage testing tool not found. Please install or update
      your PATH accordingly prior to enabling code coverage features.

      	   ])
   fi

   # ----------------------------------
   # include coverage compiler options
   # ----------------------------------

   HAVE_GCOV_TOOLS=1
   CFLAGS_GCOV="--coverage"

   # Test for C...

   if test "x$CC" != "x" ;then

     ac_coverage_save_CFLAGS="$CFLAGS"

     AC_LANG_PUSH([C])

     comp_name=`echo $CC | awk '{print $1}'`

     raw_compiler=$comp_name
     for mpicc_wrapper in mpicc hcc mpxlc_r mpxlc mpcc cmpicc; do
        if test "x$comp_name" = "x$mpicc_wrapper"; then
     	   raw_compiler=`$comp_name -show | cut -d" " -f1`
        fi
     done

     if test "x$raw_compiler" != "xgcc" ; then
   	 AC_MSG_ERROR([code coverage analysis requires gcc, not $raw_compiler])
     else
 	 CFLAGS="${CFLAGS_GCOV} ${CFLAGS}"
 	 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],[], 
	 	[AC_MSG_ERROR([unable to compile with code coverage ($CC)])])
     fi

     AC_LANG_POP([C])
   fi

   # Test for C++...

   if test "x$CXX" != "x";then

      ac_coverage_save_CXXFLAGS="$CXXFLAGS"

      AC_LANG_PUSH([C++])

      comp_name=`echo $CXX | awk '{print $1}'`

      raw_compiler=$comp_name
      for mpicxx_wrapper in mpic++ mpicxx mpiCC hcp mpxlC_r mpxlC mpCC cmpic++; do
         if test "x$comp_name" = "x$mpicxx_wrapper"; then
     	    raw_compiler=`$comp_name -show | cut -d" " -f1`
         fi
      done

      if test "x$raw_compiler" != "xg++" ; then
      	 AC_MSG_ERROR([code coverage analysis requires g++, not $raw_compiler])
      else
      	 CXXFLAGS="${CFLAGS_GCOV} ${CXXFLAGS}"
      	 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],[], 
	 	[AC_MSG_ERROR([unable to compile with code coverage ($CXX)])])
      fi

      AC_LANG_POP([C++])
   fi

fi

AC_SUBST(HAVE_GCOV_TOOLS)
AM_CONDITIONAL(CODE_COVERAGE_ENABLED,test x$HAVE_GCOV_TOOLS = x1)

])


