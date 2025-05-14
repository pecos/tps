FROM rockylinux:8
MAINTAINER Karl W. Schulz <karl@oden.utexas.edu>

# enable OpenHPC repository 
RUN yum install -y  http://repos.openhpc.community/OpenHPC/2/CentOS_8/x86_64/ohpc-release-2-1.el8.x86_64.rpm
RUN yum install -y yum-utils 

## Add some packages
RUN yum -y install make which git wget
RUN yum -y install emacs-nox
RUN yum -y install ohpc-autotools
RUN yum -y --enablerepo powertools install lmod-ohpc
#RUN yum -y install mfem-gnu9-mpich-ohpc
RUN yum -y install hypre-gnu9-mpich-ohpc
RUN yum -y install metis-gnu9-ohpc
RUN yum -y install phdf5-gnu9-mpich-ohpc
RUN yum -y install netcdf-gnu9-mpich-ohpc
RUN yum -y install petsc-gnu9-mpich-ohpc
RUN yum -y install superlu_dist-gnu9-mpich-ohpc
RUN yum -y install openblas-gnu9-ohpc
RUN yum -y install scalapack-gnu9-mpich-ohpc
RUN yum -y install ptscotch-gnu9-mpich-ohpc
RUN yum -y install lmod-defaults-gnu9-mpich-ofi-ohpc
RUN yum -y install gdb
RUN yum -y install man

RUN yum -y install gcc gcc-c++
RUN yum -y install valgrind-ohpc
RUN yum -y install boost-gnu9-mpich-ohpc

RUN yum -y install python38-pip
RUN pip3 install matplotlib

RUN yum -y install procps-ng
RUN yum -y install diffutils

#----------------------------------
# Add local install of MetaPhysicl
#----------------------------------

RUN wget https://github.com/roystgnr/MetaPhysicL/archive/refs/tags/v0.6.0.tar.gz -P /tmp
RUN cd /tmp; tar xfz v0.6.0.tar.gz
RUN cd /tmp; rm v0.6.0.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && cd /tmp/MetaPhysicL-0.6.0 \
    && ./bootstrap \
    && ./configure \
    && make \
    && make install
RUN rm -rf /tmp/MetaPhysicL-0.6.0

#---------------------------
# Add local install of MASA
#---------------------------
## ARG version=0.50.0
## RUN wget https://github.com/manufactured-solutions/MASA/releases/download/$version/masa-$version.tar.gz -P /tmp
## RUN cd /tmp; tar xfz /tmp/masa-$version.tar.gz

# Pull from Kevin's fork for new solutions..
RUN cd /tmp; git clone https://github.com/dreamer2368/MASA.git masa
RUN cd /tmp/masa; git checkout 887d5e26e3865bd6415503d62f9a557bbd3da4dc
RUN . /etc/profile.d/lmod.sh \
    && cd /tmp/masa \
    && ./bootstrap \
    && ./configure \
    && make \
    && make install
# RUN rm /tmp/masa-$version.tar.gz
RUN rm -rf /tmp/masa

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

# gslib
ENV gslib_ver="1.0.7"
RUN wget https://github.com/Nek5000/gslib/archive/refs/tags/v$gslib_ver.tar.gz -P /tmp
RUN cd /tmp; tar xvf /tmp/v$gslib_ver.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && cd /tmp/gslib-$gslib_ver \
    && make CC=mpicc CFLAGS="-O3 -fPIC" DESTDIR=/usr/local
RUN rm /tmp/v$gslib_ver.tar.gz

# MFEM (w/ gslib support)
ENV mfem_ver="4.5.2"
ENV mfem_prefix="/usr/local"
RUN wget https://github.com/mfem/mfem/archive/refs/tags/v$mfem_ver.tar.gz -P /tmp
RUN cd /tmp; tar xvf /tmp/v$mfem_ver.tar.gz

RUN . /etc/profile.d/lmod.sh \
    && module load hypre metis netcdf petsc superlu_dist \
    && cd /tmp/mfem-$mfem_ver \
    && make config \
    PREFIX=$mfem_prefix \
    CXXFLAGS="-O3 -fPIC -std=c++11" \
    MFEM_USE_MPI=YES \
    MFEM_USE_LAPACK=NO \
    HYPRE_OPT=-I$HYPRE_INC HYPRE_LIB="-L$HYPRE_LIB -lHYPRE" \
    METIS_OPT=-I$METIS_INC METIS_LIB="-L$METIS_LIB -lmetis" \
    MFEM_USE_NETCDF=YES NETCDF_OPT=-I$NETCDF_INC NETCDF_LIB="-L$NETCDF_LIB -lnetcdf" \
    MFEM_USE_PETSC=YES PETSC_OPT=-I$PETSC_INC PETSC_LIB="-L$PETSC_LIB -lpetsc" \
    MFEM_USE_SUPERLU=YES SUPERLU_OPT=-I$SUPERLU_DIST_INC SUPERLU_LIB="-L$SUPERLU_DIST_LIB -lsuperlu_dist" \
    MFEM_USE_GSLIB=YES GSLIB_OPT=-I/usr/local/include GSLIB_LIB="-L/usr/local/lib -lgs" \
    STATIC=NO SHARED=YES MFEM_DEBUG=NO \
    && make && make install

# Fix MFEM permissions
RUN find $mfem_prefix -path *mfem* -type f -exec chmod 644 -- {} +
RUN find $mfem_prefix -name *mfem* -type f -exec chmod 644 -- {} +

# Set some environment variables
ENV MFEM_DIR=$mfem_prefix
ENV GRVY_DIR=/usr/local
ENV MASA_DIR=/usr/local

# additional packages
RUN yum -y install git-lfs
RUN yum -y install aspell
RUN yum -y install cmake-ohpc
RUN yum -y install bc
RUN yum -y install clang-tools-extra
RUN yum -y install zlib-devel
RUN yum -y install libcurl-devel
RUN yum -y install python38-devel
RUN yum -y install perl-Data-Dumper
RUN yum -y install flex
RUN yum -y install bison

# build doxygen from scratch (no package... really?)
RUN wget https://www.doxygen.nl/files/doxygen-1.9.5.src.tar.gz -P /tmp
RUN cd /tmp; tar xvf /tmp/doxygen-1.9.5.src.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && module load cmake \
    && cd /tmp/doxygen-1.9.5 \
    && cmake -G "Unix Makefiles" && make && make install
RUN rm /tmp/doxygen-1.9.5.src.tar.gz

RUN yum -y install graphviz
RUN pip3 install gcovr

RUN yum -y install gsl-gnu9-ohpc

RUN . /etc/profile.d/lmod.sh \
    && pip3 install "numpy<2.0" \
    && pip3 install scipy \
    && pip3 install mpi4py

RUN pip3 install "pybind11[global]"

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig


# load certain packages by default
RUN echo "module try-add metis" >> /etc/profile.d/lmod.sh
RUN echo "module try-add hypre" >> /etc/profile.d/lmod.sh
RUN echo "module try-add phdf5" >> /etc/profile.d/lmod.sh
RUN echo "module try-add openblas" >> /etc/profile.d/lmod.sh
RUN echo "module try-add ptscotch" >> /etc/profile.d/lmod.sh
RUN echo "module try-add scalapack" >> /etc/profile.d/lmod.sh
RUN echo "module try-add superlu_dist" >> /etc/profile.d/lmod.sh
RUN echo "module try-add petsc" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add valgrind" >> /etc/profile.d/lmod.sh
RUN echo "module try-add gsl" >> /etc/profile.d/lmod.sh

# include git sha in motd
COPY motd.sh /etc/profile.d/motd.sh

## Define a user
RUN useradd -u 1000 -m test
WORKDIR /home/test
USER test
