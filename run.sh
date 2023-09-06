#!/bin/bash

cd test

EXE=../src/tps
# RUNFILE="inputs/input.malamas.test.ini"
RUNFILE="inputs/plasma.ini"
# RUNFILE="inputs/input.plasma.nlte.test.ini"


NPROC=1

mpirun -np $NPROC $EXE --runFile $RUNFILE
# mpirun -np $NPROC libtool --mode=execute gdb --args $EXE --runFile $RUNFILE
