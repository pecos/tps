ARG base_image=nvidia/cuda:12.2.2-devel-rockylinux8
#eriksf/tacc-base:rockylinux8-cuda11-tf2.11-pt1.13-mvapich2.3-ib
FROM ${base_image}
MAINTAINER Umberto Villa <uvilla@oden.utexas.edu>

ARG make_cores=6

########################################
# Configure ENV 
########################################
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

########################################
# Add docker-clean & entrypoint script
########################################
COPY extras/docker-clean /usr/bin/docker-clean
RUN chmod a+rx /usr/bin/docker-clean && docker-clean

########################################
# Add OS updates and IB stack
########################################

RUN yum install -y gcc-toolset-11-toolchain

RUN yum install -y \
    which \
    wget \
    git \ 
    git-lfs \
    file \
    && docker-clean


RUN source scl_source enable gcc-toolset-11 && which gcc
RUN source scl_source enable gcc-toolset-11 && gcc --version
RUN source scl_source enable gcc-toolset-11 && gfortran --version

RUN yum install -y \
    ca-certificates \
    libibmad-devel \
    libibumad \
    libibverbs-devel \
    librdmacm \
    libxml2-devel \
    && docker-clean

RUN yum install -y \
    boost-devel \
    cmake \
    zlib-devel \
    bzip2-devel \
    xz-devel \
    libcurl-devel \
    automake \
    libtool \
    openssl-devel \
    libffi-devel \
    bison \
    && docker-clean

RUN ldconfig

########################################
# Install mpi
########################################
# Install mvapich2-2.3
ARG MAJV=2
ARG MINV=3
ARG BV=.7
ARG DIR=mvapich${MAJV}-${MAJV}.${MINV}${BV}

RUN source scl_source enable gcc-toolset-11 && \
    curl http://mvapich.cse.ohio-state.edu/download/mvapich/mv${MAJV}/${DIR}.tar.gz | tar -xzf - \
    && cd ${DIR} \
    && ./configure \
       --with-device=ch3 \
       --with-ch3-rank-bits=32 \
       --enable-fortran=yes \
       --enable-cxx=yes \
       --enable-romio \
       #--enable-cuda \
       #--with-cuda-include=/usr/local/cuda/include \
       #--with-cuda-libpath=/usr/local/cuda/lib64 \
       --enable-fast=O3 \
       FFLAGS="-w -fallow-argument-mismatch -O2" \
    #&& make -j $(($(nproc --all 2>/dev/null || echo 2) - 2)) \
    && make -j ${make_cores} \
    && make install \
    && cd ../ && rm -rf ${DIR} \
    && rm -rf /usr/local/share/doc/mvapich2

# Add hello world
COPY extras/hello.c /tmp/hello.c
RUN source scl_source enable gcc-toolset-11 && \
    mpicc /tmp/hello.c -o /usr/local/bin/hellow \
    && rm /tmp/hello.c

# Build benchmark programs
COPY extras/install_benchmarks.sh /tmp/install_benchmarks.sh
RUN source scl_source enable gcc-toolset-11 && source /tmp/install_benchmarks.sh \
    && rm /tmp/install_benchmarks.sh

# Test installation
RUN MV2_SMP_USE_CMA=0 mpirun -n 2 hellow



RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp \
    && wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz \
    && tar xzf Python-3.12.2.tgz \
    && cd Python-3.12.2 \
    && ./configure --prefix=/opt/python/3.12.2/ --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi --enable-shared \
    && make -j $make_cores \
    && make altinstall \
    && rm /tmp/Python-3.12.2.tgz


RUN echo /opt/python/3.12.2/lib > /etc/ld.so.conf.d/python.conf
RUN ldconfig

RUN /opt/python/3.12.2/bin/python3.12 -m pip install --upgrade pip setuptools wheel \
    && ln -s /opt/python/3.12.2/bin/python3.12     /opt/python/3.12.2/bin/python3 \
    && ln -s /opt/python/3.12.2/bin/python3.12        /opt/python/3.12.2/bin/python \
    && ln -s /opt/python/3.12.2/bin/pip3.12           /opt/python/3.12.2/bin/pip3 \
    && ln -s /opt/python/3.12.2/bin/pip3.12           /opt/python/3.12.2/bin/pip \
    && ln -s /opt/python/3.12.2/bin/pydoc3.12         /opt/python/3.12.2/bin/pydoc \
    && ln -s /opt/python/3.12.2/bin/idle3.12          /opt/python/3.12.2/bin/idle \
    && ln -s /opt/python/3.12.2/bin/python3.12-config      /opt/python/3.12.2/bin/python-config
RUN alternatives --install /usr/bin/python3 python3 /opt/python/3.12.2/bin/python3 99
RUN alternatives --install /usr/bin/unversioned-python python /opt/python/3.12.2/bin/python3 99
RUN alternatives --install /usr/bin/pip3 pip3 /opt/python/3.12.2/bin/pip3 99
RUN alternatives --install /usr/bin/pip pip /opt/python/3.12.2/bin/pip 99

ENV INSTALL_DIR=/opt/sw

# MASA INSTALL
ENV MASA_DIR=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp \
    && git clone https://github.com/dreamer2368/MASA.git masa \
    && cd masa && git checkout 887d5e26e3865bd6415503d62f9a557bbd3da4dc \
    && ./bootstrap \
    && ./configure --prefix=$MASA_DIR \
    && make -j ${make_cores} \
    && make install

#GRVY INSTALL
ENV grvy_ver="0.37.0"
ENV GRVY_DIR=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget https://github.com/hpcsi/grvy/releases/download/$grvy_ver/grvy-$grvy_ver.tar.gz \
    && tar xfz grvy-$grvy_ver.tar.gz \
    && cd grvy-$grvy_ver \
    && ./configure CXXFLAGS="-std=c++11" --prefix=$GRVY_DIR  --enable-boost-headers-only \
    && make -j ${make_cores} && make install \
    && rm -rf $GRVY_DIR/lib/*.la $GRVY_DIR/lib/*.a


ENV gslib_ver="1.0.7"
ENV GSLIB_DIR=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget https://github.com/Nek5000/gslib/archive/refs/tags/v$gslib_ver.tar.gz \
    && tar xvf v$gslib_ver.tar.gz \
    && cd gslib-$gslib_ver \
    && make -j ${make_cores} CC=mpicc CFLAGS="-O3 -fPIC" DESTDIR=$GSLIB_DIR


ENV HYPRE_DIR=$INSTALL_DIR
ENV HYPRE_INC=$HYPRE_DIR/include
ENV HYPRE_LIB=$HYPRE_DIR/lib
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget  https://github.com/hypre-space/hypre/archive/refs/tags/v2.26.0.tar.gz \
    && tar -zxvf v2.26.0.tar.gz \
    && cd hypre-2.26.0/src/ \
    && ./configure --enable-shared --disable-fortran --prefix=$HYPRE_DIR \
    && make -j ${make_cores} \
    && make check \
    && make install

ENV METIS_DIR=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz \
    && tar -xvf metis-5.1.0.tar.gz \
    && cd metis-5.1.0 \
    && make config prefix=$METIS_DIR shared=1 \
    && make -j ${make_cores} && make install

ENV HDF5_DIR=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_2/source/hdf5-1.12.2.tar.gz \
    && tar -xvf hdf5-1.12.2.tar.gz \
    && cd hdf5-1.12.2 \
    && CC=mpicc ./configure --enable-parallel --prefix=$HDF5_DIR \
    && make -j ${make_cores} && make install

ARG cuda_arch=sm_80
ENV mfem_ver="4.5.2"
ENV mfem_prefix=$INSTALL_DIR
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget https://github.com/mfem/mfem/archive/refs/tags/v$mfem_ver.tar.gz \
    && tar xvf v$mfem_ver.tar.gz \
    && cd mfem-$mfem_ver \
    && unset MFEM_DIR \
    && make pcuda CUDA_ARCH=${cuda_arch} PREFIX=$mfem_prefix \
       MFEM_DEBUG=NO STATIC=NO SHARED=YES \
       HYPRE_OPT="-I$HYPRE_INC" HYPRE_LIB="-L$HYPRE_LIB -lHYPRE" \
       MFEM_USE_METIS_5=YES METIS_OPT="-I$METIS_DIR/include" METIS_LIB="-L$METIS_DIR/lib -lmetis" \
       MFEM_USE_GSLIB=YES GSLIB_OPT="-I$GSLIB_DIR/include" GSLIB_LIB="-L$GSLIB_DIR/lib -lgs" -j ${make_cores}\
       && cd examples && make -j ${make_cores} && cd .. && make install

ENV MFEM_DIR=$mfem_prefix


########################################
# Install mpi4py
########################################
RUN source scl_source enable gcc-toolset-11 && \
    pip install cython==0.29.37 psutil scikit-build nvtx && docker-clean
RUN source scl_source enable gcc-toolset-11 && \
    pip install numpy scipy  matplotlib cupy-cuda12x sympy numba multiprocess "pybind11[global]" lxcat_data_parser && docker-clean

ENV mpi4py_ver=3.1.5
RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && wget https://github.com/mpi4py/mpi4py/releases/download/$mpi4py_ver/mpi4py-$mpi4py_ver.tar.gz \
    && rm -rf  mpi4py-$mpi4py_ver && tar -zxf mpi4py-$mpi4py_ver.tar.gz && cd mpi4py-$mpi4py_ver \
    && python3 setup.py build --mpicc=$MPICC && python3 setup.py install \
    && docker-clean

RUN source scl_source enable gcc-toolset-11 && \
    cd /tmp && git clone https://github.com/ut-parla/parla-experimental \
    && cd parla-experimental \
    && git checkout 362dc07dd1df39c62f9e7b63a0698302d42e315a \
    && git submodule update --init --recursive \
    && CC=$MPICC CXX=$MPICXX make all -j ${make_cores}

RUN echo $INSTALL_DIR/lib > /etc/ld.so.conf.d/tps.conf
RUN ldconfig

RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf
RUN ldconfig

# include git sha in motd
COPY motd.sh /etc/profile.d/motd.sh

COPY extras/entry.sh /entry.sh
RUN chmod a+rx /entry.sh
ENTRYPOINT [ "/entry.sh" ]


    

