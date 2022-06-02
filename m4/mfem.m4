# SYNOPSIS
#
#   Test for MFEM
#
#   AM_PATH_MFEM( <Minimum Required Version>, <package-required=yes/no> )
#
# DESCRIPTION
#
#   Provides a --with-mfem=DIR option. Searches --with-mfem,
#   $MFEM_DIR to query provided config.mk file to ascertain MFEM
#   headers and external libraries.
#
#   On success, sets MFEM_CXXFLAGS, MFEM_LIBS, and #defines HAVE_MFEM.
#   Also defines automake conditional MFEM_ENABLED.  Assumes package
#   is optional unless overridden with $2=yes.
#
# COPYLEFT
#
#   Copyright (c) 2021 Karl W. Schulz <karl@oden.utexas.edu>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.

AC_DEFUN([AX_PATH_MFEM],
[

AC_ARG_VAR(MFEM_DIR,[root directory of MFEM installation])

AC_ARG_WITH(mfem,
  [AS_HELP_STRING([--with-mfem[=DIR]],[root directory of MFEM installation (default = MFEM_DIR)])],
  [with_mfem=$withval
if test "${with_mfem}" != yes; then
    MFEM_PREFIX=$withval
fi
],[
# assume a sensible default of --with-mfem=yes
with_mfem=yes
if test "x${MFEM_DIR}" != "x"; then
   MFEM_PREFIX=${MFEM_DIR}
fi
])

AC_ARG_ENABLE([mfem-env-vars],
         [AS_HELP_STRING([--enable-mfem-env-vars],[Allow environment variable replacement for 3rd party MFEM libs])],
	 [],[enable_mfem_env_vars=no])

# package requirement; if not specified, the default is to assume that
# the package is optional

is_package_required=ifelse([$2], ,no, $2 )

HAVE_MFEM=0

dnl snarf MFEM config
AC_MSG_CHECKING(for MFEM external library configuration)

dnl Query MFEM 3rd party libs
dnl the _ENV variant potentially has paths replaced by environment variables
MFEM_EXT_LIBS=`$srcdir/m4/snarf_mfem.py --libs --noreplace --dir $MFEM_PREFIX`
MFEM_EXT_LIBS_ENV=`$srcdir/m4/snarf_mfem.py --libs --dir $MFEM_PREFIX`
AC_MSG_RESULT($MFEM_EXT_LIBS)

dnl Query MFEM 3rd party headers
dnl the _ENV variant potentially has paths replaced by environment variables
AC_MSG_CHECKING(for MFEM header paths)
MFEM_EXT_INC=`$srcdir/m4/snarf_mfem.py --includes --noreplace --dir $MFEM_PREFIX`
MFEM_EXT_INC_ENV=`$srcdir/m4/snarf_mfem.py --includes --dir $MFEM_PREFIX`
AC_MSG_RESULT($MFEM_EXT_INC)

dnl if the user called the macro, check for package,
dnl decide what to do based on whether the package is required or not.

if test -d "${MFEM_PREFIX}/lib" ; then
   MFEM_LIBS="-L${MFEM_PREFIX}/lib -lmfem -Xlinker -rpath -Xlinker ${MFEM_PREFIX}/lib $MFEM_EXT_LIBS"
fi

if test -d "${MFEM_PREFIX}/include" ; then
    MFEM_CXXFLAGS="-I${MFEM_PREFIX}/include $MFEM_EXT_INC"
fi

ac_MFEM_save_CXXFLAGS="$CXXFLAGS"
ac_MFEM_save_LDFLAGS="$LDFLAGS"
ac_MFEM_save_LIBS="$LIBS"


dnl is this a GPU-enabled MFEM?
if test "x${CUDA_CXXFLAGS}" != "x" ; then
  CXXFLAGS="${CXXFLAGS} ${CUDA_CXXFLAGS}"
fi

if test "x${CUDA_LDFLAGS}" != "x" ; then
  LDFLAGS="${LDFLAGS} ${CUDA_LDFLAGS}"
  if test "x${OPENBLAS_LIBS}" != "x" ; then
     MFEM_LIBS="$MFEM_LIBS ${OPENBLAS_LIBS}"
  fi
fi

CXXFLAGS="${MFEM_CXXFLAGS} ${CXXFLAGS}"
LDFLAGS="${MFEM_LIBS} ${LDFLAGS}"

AC_LANG_PUSH([C++])
AC_CHECK_HEADER([mfem.hpp],[found_header=yes],[found_header=no])

dnl -----------------------
dnl Minimum version check
dnl ----------------------

min_mfem_version=ifelse([$1], ,0.29, $1)

dnl looking for major.minor.micro style versioning

MAJOR_VER=`echo $min_mfem_version | sed 's/^\([[0-9]]*\).*/\1/'`
if test "x${MAJOR_VER}" = "x" ; then
   MAJOR_VER=0
fi

MINOR_VER=`echo $min_mfem_version | sed 's/^\([[0-9]]*\)\.\{0,1\}\([[0-9]]*\).*/\2/'`
if test "x${MINOR_VER}" = "x" ; then
   MINOR_VER=0
fi

MICRO_VER=`echo $min_mfem_version | sed 's/^\([[0-9]]*\)\.\{0,1\}\([[0-9]]*\)\.\{0,1\}\([[0-9]]*\).*/\3/'`
if test "x${MICRO_VER}" = "x" ; then
   MICRO_VER=0
fi


dnl begin additional test(s) if header if available

if test "x${found_header}" = "xyes" ; then

    AC_MSG_CHECKING(for mfem - version >= $min_mfem_version)
    version_succeeded=no

    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    @%:@include <mfem/config/config.hpp>
        ]], [[
        #if MFEM_VERSION_MAJOR > $MAJOR_VER
        /* Sweet nibblets */
        #elif (MFEM_VERSION_MAJOR >= $MAJOR_VER) && (MFEM_VERSION_MINOR > $MINOR_VER)
        /* Winner winner, chicken dinner */
        #elif (MFEM_VERSION_MAJOR >= $MAJOR_VER) && (MFEM_VERSION_MINOR >= $MINOR_VER) && (MFEM_VERSION_PATCH >= $MICRO_VER)
        /* I feel like chicken tonight, like chicken tonight? */
        #else
        #  error version is too old
        #endif
    ]])],[
        AC_MSG_RESULT(yes)
        version_succeeded=yes
    ],[
        AC_MSG_RESULT(no)
    ])

    dnl do we need separate fortran linkage?

    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    @%:@include <mfem.h>
        ]], [[
        #if MFEM_MINOR_VERSION > 29
        #else
        #  error version is too old
        #endif
    ]])],[
        fortran_separate=yes
    ],[
        fortran_separate=no
    ])

   if test "$version_succeeded" != "yes";then
      if test "$is_package_required" = yes; then
         AC_MSG_ERROR([
   
   The detected MFEM library version does not meet the minimum versioning
   requirements ($min_mfem_version).  Please use --with-mfem to specify the location
   of an updated installation or consider upgrading the system version.
   
         ])
      fi
   fi

   dnl Library availability
   
   AC_MSG_CHECKING([for -lmfem linkage])
   
   #AX_CXX_CHECK_LIB([mfem], [mfem::GetVersionMinor()])
   AX_CXX_CHECK_LIB([mfem], [mfem::GetVersionMinor()],[found_library=yes],AC_MSG_ERROR([uh oh]))

fi   dnl end test if header if available

AC_LANG_POP([C++])

CXXFLAGS="$ac_MFEM_save_CXXFLAGS"
LDFLAGS="$ac_MFEM_save_LDFLAGS"
LIBS="$ac_MFEM_save_LIBS"

succeeded=no
if test "$found_header" = yes; then
    if test "$version_succeeded" = yes; then
       if test "$found_library" = yes; then
          succeeded=yes
       fi
    fi
fi

if test "$succeeded" = no; then
   if test "$is_package_required" = yes; then
      AC_MSG_ERROR([libMFEM not found.  Try either --with-mfem or setting MFEM_DIR.])
   else
      AC_MSG_NOTICE([optional MFEM library not found])
      MFEM_CXXFLAGS=""   # MFEM_CXXFLAGS empty on failure
      MFEM_LIBS=""       # MFEM_LIBS empty on failure      
   fi
else
    HAVE_MFEM=1
    AC_DEFINE(HAVE_MFEM,1,[Define if MFEM is available])
    if test x$enable_mfem_env_vars = xyes ;then
        MFEM_CXXFLAGS="-I${MFEM_PREFIX}/include $MFEM_EXT_INC_ENV"
        MFEM_LIBS="-L${MFEM_PREFIX}/lib -lmfem $MFEM_EXT_LIBS_ENV"
    fi
    AC_SUBST(MFEM_CXXFLAGS)
    AC_SUBST(MFEM_LIBS)
    AC_SUBST(MFEM_PREFIX)    
fi

AC_SUBST(HAVE_MFEM)

AM_CONDITIONAL(MFEM_ENABLED,test x$HAVE_MFEM = x1)

])
