#!/bin/bash

cd test

rm -R output-plasma
rm -R output-plasma-refine2

# cp Restart/lowP/restart_output-plasma.LTE.h5 restart_output-plasma.sol.h5
# cp Restart/highP/restart_output-plasma-refine2.LTE.h5 restart_output-plasma-refine2.sol.h5
cp Restart/highP/restart_output-plasma-refine2.NLTE.h5 restart_output-plasma-refine2.sol.h5



EXE=../src/tps
# RUNFILE="inputs/input.malamas.test.ini"
RUNFILE="inputs/plasma.ini"
# RUNFILE="inputs/input.plasma.nlte.test.ini"


NPROC=10

mpirun -np $NPROC $EXE --runFile $RUNFILE
# mpirun -np $NPROC libtool --mode=execute gdb --args $EXE --runFile $RUNFILE


