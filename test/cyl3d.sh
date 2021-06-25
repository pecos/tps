#!/usr/bin/env bats

TEST="cyl3d"
RUNFILE="input.3d.cyl"

@test "[$TEST] check for input file $RUNFILE" {
    run test -s $RUNFILE
    [[ $status -eq 0 ]]
}

@test "[$TEST] run tps with input -> $RUNFILE" {
    run ../src/tps --runFile $RUNFILE
    [[ $status -eq 0 ]]
}

