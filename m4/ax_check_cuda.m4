##### 
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA Driver API/nvcc is available, i.e. existence of:
# 	cuda.h
#   libcuda.so
#   nvcc
#
# If something isn't found, fails straight away.
#
# Locations of these are included in 
#   CUDA_CFLAGS and 
#   CUDA_LDFLAGS.
# Path to nvcc is included as
#   NVCC_PATH
# in config.h
# 
# The author is personally using CUDA such that the .cu code is generated
# at runtime, so don't expect any automake magic to exist for compile time
# compilation of .cu files.
#
# LICENCE
# Public domain
#
# AUTHOR
# wili
#
##### 

AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this		
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])

# Setting the prefix to the default if only --with-cuda was given
if test "$cuda_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		cuda_prefix="/usr/local/cuda"
	fi
fi

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -x "$cuda_prefix/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	NVCC_PATH="$cuda_prefix/bin"
else
	AC_MSG_RESULT([not found!])
	AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
fi

# We need to add the CUDA search directories for header and lib searches

# Saving the current flags
ax_save_CXXFLAGS="${CXXFLAGS}"
ax_save_LDFLAGS="${LDFLAGS}"

# Announcing the new variables
AC_SUBST([CUDA_CXXFLAGS])
AC_SUBST([CUDA_LDFLAGS])

CUDA_CXXFLAGS="-I$cuda_prefix/include"
CXXFLAGS="$CUDA_CXXFLAGS $CXXFLAGS"
CUDA_LDFLAGS="-L$cuda_prefix/lib"
LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"

# And the header and the lib
AC_LANG_PUSH([C++])
AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldn't find cuda.h]), [#include <cuda.h>])
AC_CHECK_LIB([cuda], [cuInit], [], AC_MSG_FAILURE([Couldn't find libcuda]))
AC_LANG_POP([C++])

# Returning to the original flags
CXXFLAGS=${ax_save_CXXFLAGS}
LDFLAGS=${ax_save_LDFLAGS}

])
