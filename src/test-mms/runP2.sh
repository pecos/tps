#!/bin/bash

export LD_LIBRARY_PATH=$MASA_LIB:$LD_LIBRARY_PATH

mpirun -np 4 ../MulPhyS.p -run p2r0-euler.run >& p2r0-euler.out
mpirun -np 4 ../MulPhyS.p -run p2r1-euler.run >& p2r1-euler.out
#mpirun -np 4 ../MulPhyS.p -run p2r2-euler.run >& p2r2-euler.out


