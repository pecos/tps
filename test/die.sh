#!/bin/bash
# wrapper to launch solver in background and create a DIE file while running

if [ -n "$1" ];then
    input=$1
else    
    input=input.2iters.cyl.mod
fi

timeout=90

timeout ${timeout}  mpirun -np 2 ../src/tps --runFile ${input} &
sleep 1
touch DIE
wait $!
ret=$?

# pass thru exit code of TPS
exit $ret


