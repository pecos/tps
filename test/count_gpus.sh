   #!/bin/bash

   if command -v nvidia-smi &> /dev/null; then
   NUM_GPUS=`nvidia-smi -L | wc -l`
   elif command -v rocm-smi &> /dev/null; then
   NUM_GPUS=`rocm-smi -i | grep GPU | wc -l`
   elif command -v /sbin/lspci &> /dev/null; then
     NUM_GPUS=/sbin/lspci | grep "VGA compatible controller" | grep -v $SKIP | wc -l
     if [ $NUM_GPUS -eq 0 ];then
     NUM_GPUS=`/sbin/lspci | grep "controller: NVIDIA" | wc -l`
   fi
   fi
   echo $NUM_GPUS
