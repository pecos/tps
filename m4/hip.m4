#
# SYNOPSIS
#
#   AX_HIP_CHECK()
#
# DESCRIPTION
#
#   Check if HIP-based build is desired and set build flags accordingly.


AC_DEFUN([AX_HIP_CHECK],[
    # Check for --enable-gpu-hip
    AC_ARG_ENABLE([gpu-hip],
     [AS_HELP_STRING([--enable-gpu-hip],[Enable GPU build, requires associated MFEM with HIP support.])],
     [],[enable_gpu_hip=no])

     AC_MSG_CHECKING([if requesting gpu-hip build])

     AC_ARG_VAR(HI_ARCH,"Specifies target architecture for HIP build")

     if test x$enable_gpu_hip = xyes ;then

        AC_MSG_RESULT([yes])
        AC_DEFINE([_GPU_])

        # make sure we were given a desired cuda arch
        AC_MSG_CHECKING([if gpu arch was provided])
        if test x$HIP_ARCH = x; then
           AC_MSG_RESULT([no])
           echo " "
           echo "A desired HIP_ARCH is required. Example settings include:"
           echo "HIP_ARCH=gfx803        # Zaphod"
           echo " "
           AC_MSG_ERROR([Please rerun configure with a valid HIP_ARCH setting.])
        else
            AC_MSG_RESULT([yes])
        fi

        AC_CHECK_PROG(HIPCC,[hipcc],[yes],[no])
        if test "x$HIPCC" = "xno" ; then
           AC_MSG_ERROR([hipcc must be available, please update PATH accordingly.])
        fi
        CXX=hipcc
        HIP_CXXFLAGS="-I$MPI_DIR/include"
        HIP_LDFLAGS="-L$MPI_DIR/lib -lmpi"

        ENABLE_GPU=yes
        ENABLE_HIP=yes

        AC_SUBST(HIP_CXXFLAGS)
        AC_SUBST(HIP_LDFLAGS)
        AC_DEFINE([_HIP_])

     else
        AC_MSG_RESULT([no])
        ENABLE_GPU=no
        ENABLE_HIP=no
     fi

     AM_CONDITIONAL(HIP_ENABLED,test x$ENABLE_HIP = xyes)
     AM_CONDITIONAL(GPU_ENABLED,test x$ENABLE_HIP = xyes)
])
