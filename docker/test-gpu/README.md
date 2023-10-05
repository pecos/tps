# TPS GPU Container

This container was tested using [apptainer](https://apptainer.org).

## Install apptainer

See [here](https://apptainer.org/docs/admin/main/installation.html#install-rpm-from-epel-or-fedora) for Red Hat or Fedora RPM:

```
sudo yum install -y epel-release
sudo yum install -y apptainer
```

See [here](https://apptainer.org/docs/admin/main/installation.html#install-rpm-from-epel-or-fedora) for Ubuntu packages:

```
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

## Running the GPU container with apptainer

First you need to pull the container locally
cd ..
```
apptainer pull docker://pecosut/tps_gpu_env:latest
```

This will save the apptainer image `tps_gpu_env_latest.sif` in the current directory.


Next request an interactive shell on the container (use option `--nv` to enable GPU support)

```
apptainer shell --nv ps_gpu_env_latest.sif
```

Inside apptainer, load the modules needed to build and run tps

```
Apptainer> source /etc/profile.d/lmod.sh
Apptainer> mkdir build-cuda && cd build-cuda
Apptainer> ../configure --enable-pybind11 --enable-gpu-cuda CUDA_ARCH=sm_75
```