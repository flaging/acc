# acc

acc is accelerate component of hardware, mostly based on GPU.

## cpp deps

```bash
apt-get install ninja-build
apt-get -y install cudnn9-cuda-12
wget -qO- https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh | bash
conda create -n triton python==3.12
conda activate triton
pip install -r requirements.txt
```

## kernels

[01.vector-add](./docs/01.vector-add.md)