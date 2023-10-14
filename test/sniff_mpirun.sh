#!/bin/bash

# PURPOSE understand how to run parallel jobs on HPC clusters

found=1

# defaut to mpirun if available
MPIRUN=$(type -P mpirun) || found=0

# if not, try flux
if [ $found -eq 0 ]; then
    tmp=$(type -P flux) && MPIRUN="$(type -P flux) run" || found=0
fi

# if neither mpirun or flux found, then MPIRUN is empty and tests can be skipped using this
#[ "x$MPIRUN" != "x" ] || skip "Cannot launch parallel job"

echo $MPIRUN
