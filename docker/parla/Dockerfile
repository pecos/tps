ARG base_image=pecosut/tps_gpu_env:sm_80
FROM ${base_image}
MAINTAINER Umberto Villa <uvilla@oden.utexas.edu>

USER root

ARG parla_token

RUN . /etc/profile.d/lmod.sh &&\
    pip3 install --upgrade pip &&\
    pip3 install cython &&\
    pip3 install psutil &&\
    pip3 install scikit-build &&\
    pip3 install nvtx


COPY parla-experimental parla-experimental

RUN chown -R test parla-experimental

USER test

RUN . /etc/profile.d/lmod.sh && ml gnu9 && \ 
    cd parla-experimental && \
    git checkout tps-interface && \
    sed -i 's\git@github.com:\https://github.com/\g' .gitmodules && \
    git submodule update --init --recursive &&\
    export CC=/opt/ohpc/pub/compiler/gcc/9.4.0/bin/gcc && \
    export CXX=/opt/ohpc/pub/compiler/gcc/9.4.0/bin/g++ && \
    python3 setup.py clean --all && pip3 install . --verbose

ENV PYTHONPATH=/home/test/.local/lib/python3.8/site-packages/:$PYTHONPATH
RUN chmod -R a+rX /home/test/.local
RUN chmod a+x /home/test/

RUN python3 -c "import parla"

