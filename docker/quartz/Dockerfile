FROM rockylinux:8
MAINTAINER Karl W. Schulz <karl@oden.utexas.edu>

# enable OpenHPC repository 
RUN yum install -y  http://repos.openhpc.community/OpenHPC/2/CentOS_8/x86_64/ohpc-release-2-1.el8.x86_64.rpm
RUN yum install -y yum-utils

## Add some packages
RUN yum -y install make which git git-lfs wget
RUN yum -y install emacs-nox
RUN yum -y install ohpc-autotools
RUN yum -y --enablerepo powertools install lmod-ohpc
RUN yum -y install bc
RUN yum -y install slurm-devel-ohpc
RUN yum -y install diffutils
RUN yum -y install mvapich2-psm2-gnu9-ohpc
RUN yum -y install mfem-gnu9-mvapich2-ohpc
RUN yum -y install boost-gnu9-mvapich2-ohpc
RUN yum -y install lmod-defaults-gnu9-mvapich2-ohpc
RUN yum -y install libcurl-devel
RUN yum -y install zlib-devel
RUN yum -y --enablerepo powertools install lua-devel
# dev tools
RUN yum -y install gdb 
RUN yum -y install man aspell
RUN yum -y install cmake-ohpc
RUN yum -y install python38
RUN yum -y install python38-devel
RUN yum -y install python38-pip
RUN yum -y install pybind11-devel
RUN yum -y install valgrind-ohpc
RUN yum -y install clang-tools-extra
RUN yum -y install diffutils
RUN yum -y install perl-Data-Dumper

RUN yum -y install htop
RUN yum -y install vim
RUN yum -y install curl
RUN yum -y install mlocate

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install --upgrade setuptools
RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install scipy
RUN python3.8 -m pip install pybind11

RUN . /etc/profile.d/lmod.sh && python3.8 -m pip install mpi4py

RUN yum -y install suitesparse
RUN yum -y install glpk
RUN yum -y install glpk-utils
RUN yum -y install glpk-devel
RUN python3.8 -m pip install swiglpk

RUN python3.8 -m pip install ipython
RUN python3.8 -m pip install matplotlib
RUN python3.8 -m pip install jupyter-core
RUN python3.8 -m pip install pyyaml
RUN python3.8 -m pip install cvxopt
RUN python3.8 -m pip install cvxpy
RUN python3.8 -m pip install networkx
RUN python3.8 -m pip install h5py

## Install IPOPT and its dependencies
RUN yum -y install mumps-gnu9-mvapich2-ohpc

RUN cd /tmp; git clone https://github.com/coin-or/Ipopt.git
RUN cd /tmp/Ipopt; mkdir build
RUN . /etc/profile.d/lmod.sh \
    && module load mumps scalapack openblas \
    && cd /tmp/Ipopt/build \
    && ../configure \
    --with-lapack-lflags="-L${OPENBLAS_LIB} -lopenblas" \
    --with-mumps-cflags="-I${MUMPS_INC}" \
    --with-mumps-lflags="-L${MUMPS_LIB} -L${MPI_DIR}/lib -L${SCALAPACK_DIR}/lib -lcmumps -ldmumps -lmumps_common -lsmumps -lzmumps -lscalapack" \
    && C_INCLUDE_PATH=$C_INCLUDE_PATH:$MPI_DIR/include/ CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/$MPI_DIR/include/ make -j 4 \
    && make test || true \
    && make install
RUN PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/ python3.8 -m pip install cyipopt

#---------------------------
# Add local install of GRVY
#---------------------------
ENV grvy_ver="0.37.0"
RUN wget https://github.com/hpcsi/grvy/releases/download/$grvy_ver/grvy-$grvy_ver.tar.gz -P /tmp
RUN cd /tmp; tar xfz /tmp/grvy-$grvy_ver.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && module load boost \
    && cd /tmp/grvy-$grvy_ver \
    && ./configure CXXFLAGS="-std=c++11" LDFLAGS="-Wl,-rpath,$BOOST_LIB" \
    && make \
    && make install
RUN rm /tmp/grvy-$grvy_ver.tar.gz

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig

# load mfem by default
RUN echo "module try-add mfem" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add mumps" >> /etc/profile.d/lmod.sh

# include git sha in motd
COPY motd.sh /etc/profile.d/motd.sh

# Define a user
RUN useradd -u 1000 -m test
WORKDIR /home/test
USER test
