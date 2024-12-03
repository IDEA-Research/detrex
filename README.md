# DETREX

## Original repository
The original code is available [here](https://github.com/IDEA-Research/detrex).

## Installation
The following steps will work on ICON servers (assuming cuda 12.4)
```
conda create -n detrex python=3.8 -y
conda activate detrex
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
python -m pip install -e detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
```