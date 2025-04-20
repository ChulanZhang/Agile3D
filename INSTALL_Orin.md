# Installation Guide

This guide provides step-by-step instructions for setting up the environment required for Agile3D.

## System Requirements

- NVIDIA GPU (with specific instructions for Xavier, TX2, and Orin platforms)
- Docker
- CUDA 12.2
- PyTorch 2.3.0
- Python 3.10

## Quick Start with Docker

1. Pull the Docker image:
```bash
docker pull nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel
```

2. Start a Docker container:
```bash
docker run -it --runtime nvidia --network host -v ~/fastdata:/home/data nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel
```

## Setting Up the Environment

### Basic Setup

```bash
# Create home directory
cd home
mkdir agile3d

# Install necessary tools
apt-get update
apt-get upgrade
apt install wget vim
```

### Configure Environment Variables

Add the following to `/root/.bashrc`:

```bash
# Update library paths
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"

# CUDA paths
export CUDA_PATH="/usr/local/cuda"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export CUDANN_INCLUDE_DIR="/usr/local/cuda/include"
export CUDANN_LIB_DIR="/usr/local/cuda/lib64"

# Spconv CUDA architecture settings
# Set the appropriate value for your device:
# export CUMM_CUDA_ARCH_LIST="7.2"  # for Xavier
# export CUMM_CUDA_ARCH_LIST="6.2"  # for TX2
export CUMM_CUDA_ARCH_LIST="8.7"    # for Orin
```

Apply changes with:
```bash
source /root/.bashrc
```

## Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
sh Miniconda3-latest-Linux-aarch64.sh
# When prompted, set installation path to /home/miniconda3

# Initialize conda
cd miniconda3/bin
conda init
source ~/.bashrc

# Check and update conda
conda -V
conda update conda
```

## Create Python Environment

```bash
# Create a new environment with Python 3.10
conda create -n agile3d python=3.10
conda activate agile3d
```

## Install PyTorch and TorchVision

```bash
# Install PyTorch 2.3.0
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy==1.25.0
pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl

# Install TorchVision 0.18.0
apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
pip3 install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

## Install Spconv and Dependencies

### Install Cumm

```bash
# Verify no existing installation
pip list | grep spconv
pip list | grep cumm
conda list | grep spconv
conda list | grep cumm

# Install cumm
git clone https://github.com/FindDefinition/cumm
cd ./cumm
git checkout tags/v0.5.3
# Modify version.txt to 0.4.9
echo "0.4.9" > version.txt
pip install -e .
cd ..
```

### Install Spconv

```bash
git clone https://github.com/traveller59/spconv
cd ./spconv
git checkout tags/v2.3.6
# Remove cumm from requires section in pyproject.toml
# Edit pyproject.toml with vim or your preferred editor
pip install -e .
cd ..
```

## Install Agile3D and Dependencies

```bash
# Clone the repository
git clone https://github.com/ChulanZhang/Agile3D.git agile3d
cd agile3d
python setup.py develop

# Install requirements
pip install -r requirements.txt
pip install torch-scatter
```

## Install CARL Controller Dependencies

```bash
pip install gym gymnasium

# Install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
# Force build to ensure proper compilation
CAUSAL_CONV1D_FORCE_BUILD="TRUE" python setup.py build
python setup.py install
cd ..
 
# Install mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba/
python setup.py install
cd ..

# Install tensorboard for logging
pip install tensorboard

# Install triton from source
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt # build-time dependencies
pip install -e python
cd ..
```