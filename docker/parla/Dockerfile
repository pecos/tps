ARG base_image=pecosut/tps_gpu_env:sm_80
ARG parla_token
FROM ${base_image}
MAINTAINER Umberto Villa <uvilla@oden.utexas.edu>

USER root

RUN . /etc/profile.d/lmod.sh &&\
    pip3 install cython &&\
    pip3 install psutil &&\
    pip3 install scikit-build &&\
    pip3 install nvtx

RUN . /etc/profile.d/lmod.sh && ml gnu9 && \ 
    cd /tmp && git clone https://${parla_token}@github.com/ut-parla/parla-experimental.git &&\
    cd parla-experimental && \
    sed -i 's\git@github.com:\https://github.com/\g' .gitmodules && \
    git submodule update --init --recursive &&\
    export CC=opt/ohpc/pub/compiler/gcc/9.4.0/bin/gcc && \
    export CXX=/opt/ohpc/pub/compiler/gcc/9.4.0/bin/g++ && \
    python3 setup.py clean --all && pip3 install . --verbose

USER test

