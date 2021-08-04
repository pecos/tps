FROM centos:8.3.2011
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


#---------------------------
# Add local install of MASA
#---------------------------
ARG version=0.50.0
RUN wget https://github.com/manufactured-solutions/MASA/releases/download/$version/masa-$version.tar.gz -P /tmp
RUN cd /tmp; tar xfz /tmp/masa-$version.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && cd /tmp/masa-$version \
    && ./configure --enable-fortran-interfaces \
    && make \
    && make install
RUN rm /tmp/masa-$version.tar.gz

RUN yum -y install gdb
RUN yum -y install man
RUN yum -y install python3-pip
RUN yum -y install gcc gcc-c++
RUN yum -y install python3-devel
RUN pip3 install matplotlib
RUN yum -y install valgrind-ohpc
RUN yum -y install zlib-devel
RUN yum -y install libcurl-devel

RUN yum -y install boost-gnu9-mpich-ohpc

ENV grvy_ver="0.34.0"
RUN wget https://github.com/hpcsi/grvy/releases/download/$grvy_ver/grvy-$grvy_ver.tar.gz -P /tmp
RUN cd /tmp; tar xfz /tmp/grvy-$grvy_ver.tar.gz
RUN . /etc/profile.d/lmod.sh \
    && module load boost \
    && cd /tmp/grvy-$grvy_ver \
    && ./configure CXXFLAGS="-std=c++11" LDFLAGS="-Wl,-rpath,$BOOST_LIB" \
    && make \
    && make install
RUN rm /tmp/grvy-$grvy_ver.tar.gz

# Install Bats

RUN wget https://github.com/bats-core/bats-core/archive/refs/tags/v1.3.0.tar.gz -P /tmp
RUN cd /tmp; tar xfz /tmp/v1.3.0.tar.gz
RUN cd /tmp/bats-core-1.3.0 \
    && ./install.sh /usr/local

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig

RUN yum -y install aspell
RUN yum -y install diffutils
RUN yum -y install cmake-ohpc
RUN yum -y install git-lfs
RUN yum -y install bc

# Upgrade MFEM
#RUN yum -y install http://obs.openhpc.community:82/OpenHPC:/2.1:/Factory/CentOS_8/x86_64/mfem-gnu9-mpich-ohpc-4.2-2.1.ohpc.2.1.x86_64.rpm
#
# load mfem by default
RUN echo "module try-add mfem" >>  /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add valgrind" >> /etc/profile.d/lmod.sh


## Define a user
RUN useradd -u 1000 -m test

WORKDIR /home/test
USER test
