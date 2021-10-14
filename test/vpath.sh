#!/bin/bash
#
# helper script masking as a test for vpath builds; creates soft links to
# necessary input items and utilities (bats binary, meshes, input files,
# reference solutions, etc) in the test directory.

testDir=`dirname $0`

# meshes
if [ ! -d meshes ];then
    ln -s $testDir/meshes .
fi

# inputs
if [ ! -d inputs ];then
    ln -s $testDir/inputs .
fi

# reference solutions
if [ ! -d ref_solns ];then
    ln -s $testDir/ref_solns .
fi

# necessary binaries
binaries="bats die.sh soln_differ"
for binary in $binaries; do
    if [ ! -x $binary ];then
        if [ -x $testDir/$binary ];then
           ln -s $testDir/$binary .
        fi
    fi
done
     
