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
echo GPU build enabled with CUDA. : $ENABLE_CUDA
if test "$ENABLE_CUDA" = "yes"; then
echo ' - 'CUDA_CXXFLAGS.......... : $CUDA_CXXFLAGS
echo ' - 'CUDA_LDFLAGS........... : $CUDA_LDFLAGS
fi
echo GPU build enabled with HIP.. : $ENABLE_HIP
if test "$ENABLE_HIP" = "yes"; then
echo ' - 'HIP_CXXFLAGS........... : $HIP_CXXFLAGS
echo ' - 'HIP_LDFLAGS............ : $HIP_LDFLAGS
fi
echo '-----------------------------------------------------------------------------------'

])
