# SYNOPSIS
#
#   Summarizes configuration settings.
#
#   AX_SUMMARIZE_CONFIG([, ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]]])
#
# DESCRIPTION
#
#   Outputs a summary of relevant configuration settings.
#
# LAST MODIFICATION
#
#   2021-12-16
#

AC_DEFUN([AX_SUMMARIZE_CONFIG],
[

echo
echo '------------------------------------- SUMMARY -------------------------------------'
echo Package version............. : $PACKAGE-$VERSION
echo
echo C++ compiler................ : $CXX
echo C++ compiler flags.......... : $CXXFLAGS
echo LDFLAGS..................... : $LDFLAGS
echo Install dir................. : $prefix
echo Build user.................. : $USER
echo Build host.................. : $BUILD_HOST
echo Configure date.............. : $BUILD_DATE
echo Build architecture.......... : $BUILD_ARCH
echo Git revision number......... : $BUILD_VERSION
echo
echo [3rd Party Libraries]:
echo
echo HDF5:
echo '  ' CXX flags................ : $HDF5_CFLAGS
echo '  ' LIBS..................... : $HDF5_LIBS
echo GRVY:
echo '  ' CXX flags................ : $GRVY_CFLAGS
echo '  ' LIBS..................... : $GRVY_LIBS
echo MFEM:

AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   CXX flags................ :" \
                                   --input "$MFEM_CXXFLAGS"`")
AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   LIBS..................... :" \
                                   --input "$MFEM_LIBS"`")
echo
echo [Additional Options]:
echo
echo SLURM support enabled....... : $ENABLE_SLURM
echo Valgrind available.......... : $enable_valgrind
echo MASA support enabled........ : $ENABLE_MASA
if test "x$ENABLE_MASA" = "xyes"; then
echo '  ' CXX flags................ : $MASA_CXXFLAGS
echo '  ' LIBS..................... : $MASA_LIBS
fi
echo Python interface............ : $enable_pybind11
if test "$enable_pybind11" = "yes"; then
echo '   ' CXX flags............... : $PYTHON_CPPFLAGS
fi
echo GPU build enabled with CUDA. : $ENABLE_CUDA
if test "$ENABLE_CUDA" = "yes"; then
AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   CUDA_CXXFLAGS............ :" \
                                   --input="$CUDA_CXXFLAGS"`")
AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   CUDA_LDFLAGS............. :" \
                                   --input="$CUDA_LDFLAGS"`")
fi
echo GPU build enabled with HIP.. : $ENABLE_HIP
if test "$ENABLE_HIP" = "yes"; then
AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   HIP_CXXFLAGS............. :" \
                                   --input="$HIP_CXXFLAGS"`")
AS_ECHO("`$srcdir/m4/wrap_lines.py --maxWidth 110 --first 1 --remain 31 --prefix "   HIP_LDFLAGS.............. :" \
                                   --input="$HIP_LDFLAGS"`")
fi

echo GPU build enabled for cpu... : $enable_gpu_cpu
echo '-----------------------------------------------------------------------------------'

])
