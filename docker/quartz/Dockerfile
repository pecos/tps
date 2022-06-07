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

# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig

# additional packages
RUN pip3 install numpy
RUN yum -y install pybind11-devel

# load mfem by default
RUN echo "module try-add mfem" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh

# include git sha in motd
COPY motd.sh /etc/profile.d/motd.sh

# Define a user
RUN useradd -u 1000 -m test
WORKDIR /home/test
USER test
