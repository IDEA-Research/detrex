This directory contains a few useful example scripts for detrex.

- `train_net.py`

A simpler training engine modified from [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py).

- `analyze_model.py`

Analyze FLOPs, parameters, activations of the detrex model modified from detectron2 [analyze_model.py](https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py).


- `benchmark.py`

Benchmark the training speed, inference speed or data loading speed of a given config, modified from [benchmark.py](https://github.com/facebookresearch/detectron2/blob/main/tools/benchmark.py)

Usage:
```bash
python tools/benchmark.py --config-file /path/to/config.py --task train/eval/data
```