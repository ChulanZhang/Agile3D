# Server Installation Guide for Agile3D

This guide provides step-by-step instructions for setting up Agile3D on a server environment.

## System Requirements

- NVIDIA GPU with CUDA support
- Docker
- CUDA 12.2
- Ubuntu 22.04
- PyTorch 2.1.0

## Docker Setup

1. Pull the Docker image:
```bash
docker pull nvidia/cuda:12.2.0-devel-ubuntu22.04
```

2. Run the Docker container:
```bash
docker run -it --gpus all --name mobisys2025 -v ~/data:/home/data nvidia/cuda:12.2.0-devel-ubuntu22.04 bash
```

## Basic Environment Setup

```bash
# Create home directory
cd home
mkdir agile3d
cd agile3d

# Install necessary tools
apt-get update
apt-get upgrade
apt install wget vim
```

## Install Miniconda

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts and accept the license agreement
# When prompted, confirm installation location

# Initialize conda
source ~/.bashrc

# Check installation
conda --version
```

## Create Python Environment

```bash
# Create a new environment with Python 3.10
conda create -n agile3d python=3.10
conda activate agile3d
```

## Install PyTorch and Dependencies

```bash
# Install PyTorch 2.1.0 with CUDA 12.1 support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install spconv for CUDA 12.0
pip install spconv-cu120
```

## Install Agile3D

```bash
# Clone the repository
git clone https://github.com/ChulanZhang/Agile3D.git agile3d
cd agile3d
python setup.py develop

# Install Waymo dataset tools
pip install waymo-open-dataset-tf-2-12-0
```

## Install CARL Controller Dependencies

```bash
# Install gym dependencies
pip install gym gymnasium

# Install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.0
pip install .
cd ..

# Install mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba/
pip install .
cd ..

# Install triton
pip install triton==2.2.0
```