# SYNOPSIS
#
#   Test for openBLAS
#
#   AM_PATH_OPENBLAS( <package-required=yes/no> )
#
# DESCRIPTION
#
#   Provides a --with-oepnblas=DIR option. Searches --with-openblas,
#   $OPENBLAS_DIR, and the usual places for openblas headers and libraries.
#
#   On success, sets BLAS_CFLAGS, BLAS_LIBS, and #defines HAVE_OPENBLAS.
#   Assumes package is optional unless overridden with $2=yes.
#
# COPYLEFT
#
#   Copyright (c) 2021 Karl W. Schulz <karl@oden.utexas.edu>

#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.

AC_DEFUN([AX_PATH_OPENBLAS],
[

AC_ARG_VAR(OPENBLAS_DIR,[root directory of openBLAS installation])

AC_ARG_WITH(openblas,
  [AS_HELP_STRING([--with-openblas[=DIR]],[root directory of OPENBLAS installation (default = OPENBLAS_DIR)])],
  [with_openblas=$withval
if test "${with_openblas}" != yes; then
    OPENBLAS_PREFIX=$withval
fi
],[
# assume a sensible default of --with-openblas=yes
with_openblas=yes
if test "x${OPENBLAS_DIR}" != "x"; then
   OPENBLAS_PREFIX=${OPENBLAS_DIR}
fi
])

# package requirement; if not specified, the default is to assume that
# the package is optional

is_package_required=ifelse([$1], ,no, $1 )

HAVE_OPENBLAS=0

# logic change: if the user called the macro, check for package,
# decide what to do based on whether the package is required or not.

    if test -d "${OPENBLAS_PREFIX}/lib" ; then
       OPENBLAS_LIBS="-L${OPENBLAS_PREFIX}/lib -lopenblas"
       OPENBLAS_FCFLAGS="-I${OPENBLAS_PREFIX}/lib"
    fi

    if test -d "${OPENBLAS_PREFIX}/include" ; then
        OPENBLAS_CFLAGS="-I${OPENBLAS_PREFIX}/include"
    fi

    ac_OPENBLAS_save_CFLAGS="$CFLAGS"
    ac_OPENBLAS_save_CPPFLAGS="$CPPFLAGS"
    ac_OPENBLAS_save_LDFLAGS="$LDFLAGS"
    ac_OPENBLAS_save_LIBS="$LIBS"

    CFLAGS="${OPENBLAS_CFLAGS} ${CFLAGS}"
    CPPFLAGS="${OPENBLAS_CFLAGS} ${CPPFLAGS}"
    LDFLAGS="${OPENBLAS_LIBS} ${LDFLAGS}"
    AC_LANG_PUSH([C])
    AC_CHECK_HEADER([cblas.h],[found_header=yes],[found_header=no])

    # begin additional test(s) if header if available

    if test "x${found_header}" = "xyes" ; then

    # Library availability

    AC_MSG_CHECKING([for -lopenblas linkage])

    AC_CHECK_LIB([openblas],cblas_dgemm,[found_library=yes],[found_library=no])

    fi   dnl end test if header if available

    AC_LANG_POP([C])

    CFLAGS="$ac_OPENBLAS_save_CFLAGS"
    CPPFLAGS="$ac_OPENBLAS_save_CPPFLAGS"
    LDFLAGS="$ac_OPENBLAS_save_LDFLAGS"
    LIBS="$ac_OPENBLAS_save_LIBS"

    succeeded=no
    if test "$found_header" = yes; then
       if test "$found_library" = yes; then
          succeeded=yes
       fi
    fi

    if test "$succeeded" = no; then
       if test "$is_package_required" = yes; then
          AC_MSG_ERROR([libOPENBLAS not found.  Try either --with-openblas or setting OPENBLAS_DIR.])
       else
          AC_MSG_NOTICE([optional OPENBLAS library not found])
          OPENBLAS_CFLAGS=""   # OPENBLAS_CFLAGS empty on failure
          OPENBLAS_FCFLAGS=""  # OPENBLAS_FCFLAGS empty on failure
          OPENBLAS_LIBS=""     # OPENBLAS_LIBS empty on failure
       fi
    else
        HAVE_OPENBLAS=1
        AC_DEFINE(HAVE_OPENBLAS,1,[Define if OPENBLAS is available])
        AC_SUBST(OPENBLAS_CFLAGS)
        AC_SUBST(OPENBLAS_FCFLAGS)
        AC_SUBST(OPENBLAS_LIBS)
        AC_SUBST(OPENBLAS_PREFIX)
    fi

    AC_SUBST(HAVE_OPENBLAS)

# fi

AM_CONDITIONAL(OPENBLAS_ENABLED,test x$HAVE_OPENBLAS = x1)

])
