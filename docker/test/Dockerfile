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
RUN yum -y install mfem-gnu9-mpich-ohpc
RUN yum -y install lmod-defaults-gnu9-mpich-ofi-ohpc
RUN yum -y install gdb
RUN yum -y install man

RUN yum -y install gcc gcc-c++
RUN yum -y install valgrind-ohpc
RUN yum -y install boost-gnu9-mpich-ohpc

RUN yum -y install python38-pip
RUN pip3 install matplotlib
RUN pip3 install pybind11
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

# additional packages
RUN yum -y install git-lfs
RUN yum -y install aspell
RUN yum -y install cmake-ohpc
RUN yum -y install bc
RUN yum -y install clang-tools-extra
RUN yum -y install zlib-devel
RUN yum -y install libcurl-devel
RUN yum -y install python3-pybind11

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig


# load certain packages by default
RUN echo "module try-add mfem" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add valgrind" >> /etc/profile.d/lmod.sh

## Define a user
RUN useradd -u 1000 -m test
WORKDIR /home/test
USER test
