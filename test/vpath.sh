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
binaries="bats die.sh soln_differ ../src/tps.py"
for binary in $binaries; do
    if [ ! -x $binary ];then
        if [ -x $testDir/$binary ];then
           ln -s $testDir/$binary $binary
        fi
    fi
done

# necessary text files
files="valgrind.suppressions"
for file in $files; do
    if [ ! -s $file ];then
        if [ -s $testDir/$file ];then
           ln -s $testDir/$file .
        fi
    fi
done

# necessary libs
libs="libtps.la"
for lib in $libs; do
    if [ ! -s $lib ];then
        if [ -s $testDir/$lib ];then
           ln -s $testDir/$lib .
        fi
    fi
done
