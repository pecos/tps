AC_DEFUN([AX_PATH_MPI4PY],
[

AC_ARG_VAR(MPI4PY_DIR,[root directory of MPI4PY installation])

AC_ARG_WITH(mpi4py,
  [AS_HELP_STRING([--with-mpi4py[=DIR]],[root directory of MPI4PY installation (default = MPI4PY_DIR)])],
  [with_MPI4PY=$withval
if test "${with_MPI4PY}" != yes; then
    MPI4PY_PREFIX=$withval
fi
],[
# assume a sensible default of --with-MPI4PY=yes
with_MPI4PY=yes
if test "x${MPI4PY_DIR}" != "x"; then
   MPI4PY_PREFIX=${MPI4PY_DIR}
fi
])

AC_ARG_ENABLE([MPI4PY-env-vars],
         [AS_HELP_STRING([--enable-MPI4PY-env-vars],[Allow environment variable replacement for 3rd party MPI4PY libs])],
	 [],[enable_MPI4PY_env_vars=no])

# package requirement; if not specified, the default is to assume that
# the package is optional

is_package_required=ifelse([$2], ,no, $2 )

HAVE_MPI4PY=0
AC_MSG_CHECKING([if mpi4py is available])
   AS_IF([$PYTHON -m mpi4py.bench helloworld >& /dev/null], [found_mpi4py=yes], [found_mpi4py=no])
   if test "x${found_mpi4py}" = "xno" ; then
      AC_MSG_RESULT([no])
      # for now, just notify user, but in future this failure will be an error
      #AC_MSG_ERROR([mpi4py not found. Please verify mpi4py is installed locally.])
      MPI4PY_CXXFLAGS=""   # MPI4PY_CXXFLAGS empty on failure
      MPI4PY_LIBS=""       # MPI4PY_LIBS empty on failure  
   else
      AC_MSG_RESULT([yes])
      HAVE_MPI4PY=1
      AC_DEFINE(HAVE_MPI4PY,1,[Define if  MPI4PY is available])
      dnl snarf MPI4PY config
      AC_MSG_CHECKING(for MPI4PY dynamic lib)
      MPI4PY_LIBS=`$srcdir/m4/snarf_mpi4py.py --lib`
      AC_MSG_RESULT($MPI4PY_LIBS)
      AC_MSG_CHECKING(for MPI4PY header paths)
      MPI4PY_INC_DIR=`$srcdir/m4/snarf_mpi4py.py --include`
      AC_MSG_RESULT($MPI4PY_INC_DIR)
      MPI4PY_CXXFLAGS="-I${MPI4PY_INC_DIR}"
   fi


AC_SUBST(HAVE_MPI4PY)
AC_SUBST(MPI4PY_CXXFLAGS)
AC_SUBST(MPI4PY_CPPFLAGS)
AC_SUBST(MPI4PY_LIBS)

CXXFLAGS="${MPI4PY_CXXFLAGS} ${CXXFLAGS}"
CPPFLAGS="${MPI4PY_CXXFLAGS} ${CPPFLAGS}"

ac_MPI4PY_save_CXXFLAGS="$CXXFLAGS"
ac_MPI4PY_save_CPPFLAGS="$CPPFLAGS"

])