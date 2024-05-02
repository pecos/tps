ARG base_image=rockylinux:8.7
FROM ${base_image}
MAINTAINER Umberto Villa <uvilla@oden.utexas.edu>

ARG make_cores=6
ARG ROCM_VERSION=5.4.1
ARG AMDGPU_VERSION=5.4.1

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

RUN dnf install -y \
    which \
    wget \
    git \ 
    git-lfs \
    file \
    && docker-clean

RUN dnf group -y install "Development Tools" && docker-clean

RUN dnf install -y \
    ca-certificates \
    libibmad-devel \
    libibumad \
    libibverbs-devel \
    librdmacm \
    libxml2-devel \
    && docker-clean

RUN dnf install -y \
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
# Install rocm
########################################
# Enable epel-release repositories
RUN dnf --enablerepo=extras install -y epel-release && docker-clean
#RUN yum --nogpgcheck install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
RUN dnf config-manager --set-enabled powertools && docker-clean

RUN dnf install -y libdrm && docker-clean
RUN dnf clean all
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/rhel8/$ROCM_VERSION/main\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo
RUN echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/rhel/8.7/main/x86_64\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/amdgpu.repo

RUN dnf install -y rocm-dev hip-devel rocm-hip-runtime-devel rocm-libs rocm-hip-libraries && docker-clean
ENV PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib::${LD_LIBRARY_PATH}
RUN which hipcc

########################################
# Install mpi
########################################


# See https://docs.archer2.ac.uk/user-guide/containers/
RUN curl -sSLO http://www.mpich.org/static/downloads/3.4.2/mpich-3.4.2.tar.gz \
   && tar -xzf mpich-3.4.2.tar.gz -C /root \
   && cd /root/mpich-3.4.2 \
   && ./configure --prefix=/usr --with-device=ch4:ofi --disable-fortran \
   && make -j ${make_cores} install \
   && rm -rf /root/mpich-3.4.2 \
   && rm /mpich-3.4.2.tar.gz

# Add hello world
COPY extras/hello.c /tmp/hello.c
RUN mpicc /tmp/hello.c -o /usr/local/bin/hellow \
    && rm /tmp/hello.c

# Build benchmark programs
COPY extras/install_benchmarks.sh /tmp/install_benchmarks.sh
RUN source /tmp/install_benchmarks.sh \
    && rm /tmp/install_benchmarks.sh

# Test installation
RUN MV2_SMP_USE_CMA=0 mpirun -n 2 hellow



RUN cd /tmp \
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
RUN cd /tmp \
    && git clone https://github.com/dreamer2368/MASA.git masa \
    && cd masa && git checkout 887d5e26e3865bd6415503d62f9a557bbd3da4dc \
    && ./bootstrap \
    && ./configure --prefix=$MASA_DIR \
    && make -j ${make_cores} \
    && make install

#GRVY INSTALL
RUN dnf install -y gcc-gfortran
ENV grvy_ver="0.37.0"
ENV GRVY_DIR=$INSTALL_DIR
RUN cd /tmp && wget https://github.com/hpcsi/grvy/releases/download/$grvy_ver/grvy-$grvy_ver.tar.gz \
    && tar xfz grvy-$grvy_ver.tar.gz \
    && cd grvy-$grvy_ver \
    && ./configure CXXFLAGS="-std=c++11" --prefix=$GRVY_DIR --disable-fortran --enable-boost-headers-only \
    && make -j ${make_cores} && make install \
    && rm -rf $GRVY_DIR/lib/*.la $GRVY_DIR/lib/*.a


ENV gslib_ver="1.0.7"
ENV GSLIB_DIR=$INSTALL_DIR
RUN cd /tmp && wget https://github.com/Nek5000/gslib/archive/refs/tags/v$gslib_ver.tar.gz \
    && tar xvf v$gslib_ver.tar.gz \
    && cd gslib-$gslib_ver \
    && make -j ${make_cores} CC=mpicc CFLAGS="-O3 -fPIC" DESTDIR=$GSLIB_DIR


ENV HYPRE_DIR=$INSTALL_DIR
ENV HYPRE_INC=$HYPRE_DIR/include
ENV HYPRE_LIB=$HYPRE_DIR/lib
RUN cd /tmp && wget  https://github.com/hypre-space/hypre/archive/refs/tags/v2.26.0.tar.gz \
    && tar -zxvf v2.26.0.tar.gz \
    && cd hypre-2.26.0/src/ \
    && ./configure --enable-shared --disable-fortran --prefix=$HYPRE_DIR \
    && make -j ${make_cores} \
    && make check \
    && make install

ENV METIS_DIR=$INSTALL_DIR
RUN cd /tmp && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz \
    && tar -xvf metis-5.1.0.tar.gz \
    && cd metis-5.1.0 \
    && make config prefix=$METIS_DIR shared=1 \
    && make -j ${make_cores} && make install

ENV HDF5_DIR=$INSTALL_DIR
RUN cd /tmp && wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_2/source/hdf5-1.12.2.tar.gz \
    && tar -xvf hdf5-1.12.2.tar.gz \
    && cd hdf5-1.12.2 \
    && CC=mpicc ./configure --enable-parallel --prefix=$HDF5_DIR \
    && make -j ${make_cores} && make install

ARG hip_arch=gfx90a
ENV mfem_ver="4.5.2"
ENV mfem_prefix=$INSTALL_DIR
ENV HIP_DIR=/opt/rocm
RUN cd /tmp && wget https://github.com/mfem/mfem/archive/refs/tags/v$mfem_ver.tar.gz \
    && tar xvf v$mfem_ver.tar.gz \
    && cd mfem-$mfem_ver \
    && unset MFEM_DIR \
    && make phip HIP_ARCH=${hip_arch} HIP_DIR=/opt/rocm PREFIX=$mfem_prefix \
       MFEM_DEBUG=NO STATIC=NO SHARED=YES \
       HYPRE_OPT="-I$HYPRE_INC" HYPRE_LIB="-L$HYPRE_LIB -lHYPRE" \
       MFEM_USE_METIS_5=YES METIS_OPT="-I$METIS_DIR/include" METIS_LIB="-L$METIS_DIR/lib -lmetis" \
       MFEM_USE_GSLIB=YES GSLIB_OPT="-I$GSLIB_DIR/include" GSLIB_LIB="-L$GSLIB_DIR/lib -lgs" -j ${make_cores}\
       && cd examples && make -j ${make_cores} && cd .. && make install

ENV MFEM_DIR=$mfem_prefix


########################################
# Install mpi4py
########################################
RUN alternatives --set python3 /opt/python/3.12.2/bin/python3
RUN alternatives --set pip /opt/python/3.12.2/bin/pip

RUN pip install cython==0.29.37 psutil scikit-build && docker-clean
RUN pip install numpy scipy  matplotlib sympy numba multiprocess "pybind11[global]" lxcat_data_parser && docker-clean

ENV mpi4py_ver=3.1.5
RUN cd /tmp && wget https://github.com/mpi4py/mpi4py/releases/download/$mpi4py_ver/mpi4py-$mpi4py_ver.tar.gz \
    && rm -rf  mpi4py-$mpi4py_ver && tar -zxf mpi4py-$mpi4py_ver.tar.gz && cd mpi4py-$mpi4py_ver \
    && MPICC=/usr/local/bin/mpicc python3 setup.py install \
    && docker-clean

RUN cd /tmp && git clone https://github.com/ut-parla/parla-experimental \
    && cd parla-experimental \
    && git checkout 362dc07dd1df39c62f9e7b63a0698302d42e315a \
    && git submodule update --init --recursive \
    && CC=$MPICC CXX=$MPICXX make all -j ${make_cores}

ENV HCC_AMDGPU_TARGET=$hip_arch
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
ARG cupy_ver=12.0.0
RUN export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH && \
    pip install cupy==$cupy_ver

RUN python3 -c "import cupy"

RUN echo $INSTALL_DIR/lib > /etc/ld.so.conf.d/tps.conf
RUN ldconfig

RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf
RUN ldconfig

COPY extras/entry.sh /entry.sh
RUN chmod a+rx /entry.sh
ENTRYPOINT [ "/entry.sh" ]
