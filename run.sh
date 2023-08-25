#!/bin/bash

cd test

EXE=../src/tps
RUNFILE="inputs/input.malamas.test.ini"
# RUNFILE="inputs/plasma.ini"


NPROC=2

mpirun -np $NPROC $EXE --runFile $RUNFILE


