#!/bin/bash

# sudo docker run --user 1001 -it -v ${HOME}:/home/test pecosut/tps_env:latest bash -l
# ./bootstrap
# ./configure --enable-pybind11 CXXFLAGS="-Wall -Werror"
# make -j 10;

cd test

rm -R output-plasma
rm -R output-plasma-refine2

# cp Restart/lowP/restart_output-plasma.LTE.h5 restart_output-plasma.sol.h5
# cp Restart/highP/restart_output-plasma-refine2.LTE.h5 restart_output-plasma-refine2.sol.h5
# cp Restart/highP/restart_output-plasma-refine2.NLTE.h5 restart_output-plasma-refine2.sol.h5

cp ../../tps-inputs/axisymmetric/argon/lowP/lte/restart_output-plasma.sol.h5 ./ 


EXE=../src/tps
RUNFILE="inputs/plasma.ini"
# RUNFILE="inputs/input.4iters.cyl.ini"

# RUNFILE="inputs/perfectGas.air.ini"

NPROC=10

mpirun -np $NPROC $EXE --runFile $RUNFILE
# mpirun -np $NPROC libtool --mode=execute gdb --args $EXE --runFile $RUNFILE


