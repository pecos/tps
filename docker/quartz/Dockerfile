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
RUN yum -y install hypre-gnu9-mvapich2-ohpc
RUN yum -y install metis-gnu9-ohpc
RUN yum -y install phdf5-gnu9-mvapich2-ohpc
RUN yum -y install netcdf-gnu9-mvapich2-ohpc
#RUN yum -y install petsc-gnu9-mvapich2-ohpc
RUN yum -y install superlu_dist-gnu9-mvapich2-ohpc
RUN yum -y install openblas-gnu9-ohpc
RUN yum -y install scalapack-gnu9-mvapich2-ohpc
RUN yum -y install ptscotch-gnu9-mvapich2-ohpc
RUN yum -y install boost-gnu9-mvapich2-ohpc
RUN yum -y install lmod-defaults-gnu9-mvapich2-ohpc
RUN yum -y install libcurl-devel
RUN yum -y install zlib-devel
RUN yum -y --enablerepo powertools install lua-devel
# dev tools
RUN yum -y install gdb 
RUN yum -y install man aspell
RUN yum -y install python3-pip
RUN yum -y install python3-devel
RUN yum -y install valgrind-ohpc
RUN yum -y install clang-tools-extra
RUN yum -y install diffutils

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


#----------------------------
# Add local install of gslib
#----------------------------
ENV gslib_ver="1.0.7"
RUN wget https://github.com/Nek5000/gslib/archive/refs/tags/v$gslib_ver.tar.gz -P /tmp
RUN cd /tmp; tar xvf /tmp/v$gslib_ver.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && cd /tmp/gslib-$gslib_ver \
    && make CC=mpicc CFLAGS="-O3 -fPIC" DESTDIR=/usr/local
RUN rm /tmp/v$gslib_ver.tar.gz


#----------------------------
# Add local install of mfem
# (to get gslib support)
#----------------------------
# This mfem corresponds to the master branch as of July 30, 2022
#
# We use this version rather than the 4.4 release because it has
# support for GSLIB interpolation with mixed meshes, which was landed
# in PR 2948 (https://github.com/mfem/mfem/pull/2948) on May 30.

ENV mfem_prefix="/usr/local"
RUN cd /tmp; git clone https://github.com/mfem/mfem.git
RUN cd /tmp/mfem; git checkout fcf50aae53db688bc22ed7c230cfbfae2d05ebaf

RUN . /etc/profile.d/lmod.sh \
    && module load mvapich2 hypre metis netcdf superlu_dist \
    && cd /tmp/mfem \
    && make config \
    PREFIX=$mfem_prefix \
    CXXFLAGS="-O3 -fPIC -std=c++11" \
    MFEM_USE_MPI=YES \
    MFEM_USE_LAPACK=NO \
    HYPRE_OPT=-I$HYPRE_INC HYPRE_LIB="-L$HYPRE_LIB -lHYPRE" \
    METIS_OPT=-I$METIS_INC METIS_LIB="-L$METIS_LIB -lmetis" \
    MFEM_USE_NETCDF=YES NETCDF_OPT=-I$NETCDF_INC NETCDF_LIB="-L$NETCDF_LIB -lnetcdf" \
    MFEM_USE_PETSC=NO \
    MFEM_USE_SUPERLU=YES SUPERLU_OPT=-I$SUPERLU_DIST_INC SUPERLU_LIB="-L$SUPERLU_DIST_LIB -lsuperlu_dist" \
    MFEM_USE_GSLIB=YES GSLIB_OPT=-I/usr/local/include GSLIB_LIB="-L/usr/local/lib -lgs" \
    STATIC=NO SHARED=YES MFEM_DEBUG=NO \
    && make && make install

RUN yum -y install findutils

# Fix MFEM permissions
RUN find $mfem_prefix -path *mfem* -type f -exec chmod 644 -- {} +
RUN find $mfem_prefix -name *mfem* -type f -exec chmod 644 -- {} +

ENV MFEM_DIR=$mfem_prefix

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig

# additional packages
RUN pip3 install numpy
RUN yum -y install pybind11-devel
RUN yum -y install python38-devel
RUN yum -y install perl-Data-Dumper

# load mfem by default
RUN echo "module try-add metis" >> /etc/profile.d/lmod.sh
RUN echo "module try-add hypre" >> /etc/profile.d/lmod.sh
RUN echo "module try-add phdf5" >> /etc/profile.d/lmod.sh
RUN echo "module try-add openblas" >> /etc/profile.d/lmod.sh
RUN echo "module try-add ptscotch" >> /etc/profile.d/lmod.sh
RUN echo "module try-add scalapack" >> /etc/profile.d/lmod.sh
RUN echo "module try-add superlu_dist" >> /etc/profile.d/lmod.sh
RUN echo "module try-add netcdf" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add valgrind" >> /etc/profile.d/lmod.sh

# include git sha in motd
COPY motd.sh /etc/profile.d/motd.sh

# Define a user
RUN useradd -u 1000 -m test
WORKDIR /home/test
USER test
