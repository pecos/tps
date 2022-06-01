#!/bin/bash

TPS_OBJS="averaging_and_rms.o faceGradientIntegration.o M2ulPhyS.o rhs_operator.o wallBC.o BCintegrator.o face_integrator.o riemann_solver.o BoundaryCondition.o fluxes.o masa_handler.o run_configuration.o domain_integrator.o forcing_terms.o mpi_groups.o sbp_integrators.o equation_of_state.o transport_properties.o inletBC.o outletBC.o utils.o io.o dgNonlinearForm.o gradients.o gradNonLinearForm.o quasimagnetostatic.o tps.o chemistry.o reaction.o collision_integrals.o argon_transport.o source_term.o ../utils/mfem_extras/pfem_extras.o"

EXTRA_OBJS="/usr/lib/../lib64/crti.o /opt/ohpc/pub/compiler/gcc/9.3.0/lib/gcc/x86_64-pc-linux-gnu/9.3.0/crtbeginS.o"

NVCC=/usr/local/cuda/bin/nvcc
CUDA_CXXFLAGS="--expt-extended-lambda -arch=sm_75 -ccbin mpicxx"
CUDA_LDFLAGS="--expt-extended-lambda -arch=sm_75 -Xcompiler=-fPIC"

MPICXX=mpic++
CXXFLAGS="-std=c++11 -g -O2"
LDFLAGS="-fPIC -shared -nostdlib"
LDEXTRA="-Xlinker -rpath -Xlinker /opt/ohpc/pub/compiler/gcc/9.3.0/lib/../lib64 -Xlinker -rpath -Xlinker /opt/ohpc/pub/compiler/gcc/9.3.0/lib/../lib64 -L/home/karl/sw/mfem-gpu-4.4//lib -lmfem -L/opt/ohpc/pub/libs/gnu9/mpich/hypre/2.18.1/lib -lHYPRE -L/opt/ohpc/pub/libs/gnu9/metis/5.1.0/lib -lmetis -lrt -L/opt/ohpc/pub/libs/gnu9/openblas/0.3.7/lib -lopenblas -L/opt/ohpc/pub/libs/gnu9/hdf5/1.10.6/lib -lhdf5 -L/opt/ohpc/pub/libs/gnu9/mpich/grvy/0.37.0/lib -lgrvy -L/usr/local/cuda/lib64 -lcudart -lcusparse -lcuda -L/opt/ohpc/pub/mpi/mpich-ofi-gnu9-ohpc/3.4.2/lib -L/opt/ohpc/pub/compiler/gcc/9.3.0/lib/gcc/x86_64-pc-linux-gnu/9.3.0 -L/opt/ohpc/pub/compiler/gcc/9.3.0/lib/gcc/x86_64-pc-linux-gnu/9.3.0/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/opt/ohpc/pub/compiler/gcc/9.3.0/lib/gcc/x86_64-pc-linux-gnu/9.3.0/../../.. -lmpicxx -lmpi /opt/ohpc/pub/compiler/gcc/9.3.0/lib/../lib64/libstdc++.so -lm -lc -lgcc_s /opt/ohpc/pub/compiler/gcc/9.3.0/lib/gcc/x86_64-pc-linux-gnu/9.3.0/crtendS.o /usr/lib/../lib64/crtn.o  -g -O2 -Xlinker -rpath -Xlinker /home/karl/sw/mfem-gpu-4.4//lib   -Xlinker -soname -Xlinker libtps.so"

LDFINAL="-Xlinker -rpath -Xlinker /home/karl/sw/mfem-gpu-4.4//lib -L/home/karl/sw/mfem-gpu-4.4//lib -L/opt/ohpc/pub/libs/gnu9/mpich/hypre/2.18.1/lib -L/opt/ohpc/pub/libs/gnu9/metis/5.1.0/lib -L/opt/ohpc/pub/libs/gnu9/openblas/0.3.7/lib -L/usr/local/cuda/lib64 libtps.so -L/opt/ohpc/pub/libs/gnu9/hdf5/1.10.6/lib -L/opt/ohpc/pub/libs/gnu9/mpich/grvy/0.37.0/lib -lmfem -lHYPRE -lmetis -lrt -lopenblas -lhdf5 -lgrvy -lcudart -lcusparse /opt/ohpc/pub/compiler/gcc/9.3.0/lib/../lib64/libstdc++.so -lm -lcuda -Xlinker -rpath -Xlinker /opt/ohpc/pub/compiler/gcc/9.3.0/lib/../lib64"


# Step 0: run usual ./bootstrap; configure; make.  This will fail at
# the final link with many errors like the following
#
# /usr/bin/ld: ./.libs/libtps.so: undefined reference to `__cudaRegisterLinkedBinary_38_tmpxft_000056c4_00000000_7_tps_cpp1_ii_17660025'
#

cd src

# start fresh
rm -rf tmp_cuda_link.o
rm -rf libtps.so libtps.so.*
rm -rf tps

# Step 1: device link
# Note: No analog of this step in current autotools-based process
$NVCC $CXXFLAGS $TPS_OBJS -dlink $CUDA_LDFLAGS -o tmp_cuda_link.o

# Step 2: build libtps
# Note: Need to modify autotools-based process to include extra object generated above
$MPICXX $LDFLAGS $EXTRA_OBJS ./tmp_cuda_link.o $TPS_OBJS $LDEXTRA -o libtps.so

# Step 3: build tps
# Note: Should be analogous to autotools-based build if above goes well
$NVCC $CXXFLAGS $CUDA_CXXFLAGS -o tps main.o  -L. libtps.so $LDFINAL

# Step 4: confirm we can run something
cd ../test
../src/tps -run inputs/input.dtconst.cyl.ini
