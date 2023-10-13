ARG base_image=pecosut/tps_cuda_base:latest
FROM ${base_image}
MAINTAINER Umberto Villa <uvilla@oden.utexas.edu>

USER root

ARG cuda_arch_number=75

# Hypre (w/ CUDA support)
RUN . /etc/profile.d/lmod.sh && module load openblas \
    && cd /tmp \
    && wget  https://github.com/hypre-space/hypre/archive/refs/tags/v2.26.0.tar.gz \
    && tar -zxvf v2.26.0.tar.gz \
    && cd hypre-2.26.0/src/ \
    && ./configure --enable-shared --with-cuda --enable-unified-memory --with-gpu-arch="${cuda_arch_number}" --disable-fortran --prefix=/usr/local \
    && make -j 6 \
    && make install

ENV HYPRE_INC=/usr/local/include
ENV HYPRE_LIB=/usr/local/lib

# MFEM (w/ gslib support)
ENV mfem_ver="4.5.2"
ENV mfem_prefix="/usr/local"
RUN wget https://github.com/mfem/mfem/archive/refs/tags/v$mfem_ver.tar.gz -P /tmp
RUN cd /tmp; tar xvf /tmp/v$mfem_ver.tar.gz

RUN . /etc/profile.d/lmod.sh \
    && module load  metis \
    && module load openblas \
    && cd /tmp/mfem-$mfem_ver \
    && make pcuda CUDA_ARCH=sm_${cuda_arch_number} \
    PREFIX=$mfem_prefix \
    STATIC=NO SHARED=YES \
    HYPRE_OPT=-I$HYPRE_INC HYPRE_LIB="-L$HYPRE_LIB -lHYPRE" \
    MFEM_USE_METIS_5=YES \
    METIS_OPT=-I$METIS_INC METIS_LIB="-L$METIS_LIB -lmetis" \
    MFEM_USE_GSLIB=YES GSLIB_OPT=-I/usr/local/include GSLIB_LIB="-L/usr/local/lib -lgs" -j 6 \
    && make install

# Fix MFEM permissions
RUN find $mfem_prefix -path *mfem* -type f -exec chmod 644 -- {} +
RUN find $mfem_prefix -name *mfem* -type f -exec chmod 644 -- {} +

# Set some environment variables
ENV MFEM_DIR=$mfem_prefix
ENV GRVY_DIR=/usr/local
ENV MASA_DIR=/usr/local

# We need to install gls after mfem, because mfem requires a different type of GSL lib
RUN yum -y install gsl-gnu9-ohpc


# Register new libs installed into /usr/local/lib with linker
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/class.conf
RUN ldconfig


# load certain packages by default
RUN echo "module try-add metis" >> /etc/profile.d/lmod.sh
RUN echo "module try-add phdf5" >> /etc/profile.d/lmod.sh
RUN echo "module try-add petsc" >> /etc/profile.d/lmod.sh
RUN echo "module try-add openblas" >> /etc/profile.d/lmod.sh
RUN echo "module try-add boost" >> /etc/profile.d/lmod.sh
RUN echo "module try-add valgrind" >> /etc/profile.d/lmod.sh
RUN echo "module try-add gsl" >> /etc/profile.d/lmod.sh

ENV CUDA_ARCH=sm_${cuda_arch_number}

USER test
