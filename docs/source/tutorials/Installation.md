# Installation
**detrex** provides an editable installation way for you to develop your own project based on detrex's framework.

## Requirements
- Linux with Python ≥ **3.7**
- PyTorch ≥ **1.9** and [torchvision](https://github.com/pytorch/vision/) matches the PyTorch insallation. Install them following the official instructions from [pytorch.org](https://pytorch.org) to make sure of this.


## Build detrex from Source
gcc & g++ ≥ 5.4 are required as [detectron2](https://github.com/facebookresearch/detectron2), [ninja](https://ninja-build.org/) is optional but recommended for faster build. After having them, install detrex as follows:

- Firstly, create a conda virtual environment named `detrex` and activate it
```bash
$ conda create -n detrex python=3.7 -y
$ conda activate detrex
```
- Secondly, clone `detrex` and initialize the `detectron2` submodule.
```bash
$ git clone https://github.com/IDEACVR/detrex.git
$ cd detrex
$ git submodule init
$ git submodule update
```
- Finally, install `detectron2` and build an editable version of `detrex` for better usage.

```bash
$ python -m pip install -e detectron2
$ pip install -e .
```

To **rebuild** detrex that's built from a local clone, use `rm -rf build/` to clean the old build first.

## Common Installation Issues
If you meet some installation problems with `detectron2`, please see [detectron2 installation issues](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#common-installation-issues) for more details.

Click each issue for its solutions:
<details>
<summary> NotImplementedError: Cuda is not availabel </summary>

If you're running with `slurm`, make sure that [CUDA runtime](https://developer.nvidia.com/cuda-downloads) has been installed. Please specify the environment `CUDA_HOME` to the path of `CUDA` dir, e.g., `CUDA_HOME=/usr/local/cuda-11.3` which is the default path to the installed CUDA runtime.

</details>

