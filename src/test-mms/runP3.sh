#!/bin/bash

export LD_LIBRARY_PATH=$MASA_LIB:$LD_LIBRARY_PATH

mpirun -np 4 ../MulPhyS.p -run p3r0-euler.run >& p3r0-euler.out
mpirun -np 4 ../MulPhyS.p -run p3r1-euler.run >& p3r1-euler.out
#mpirun -np 4 ../MulPhyS.p -run p3r2-euler.run >& p3r2-euler.out


