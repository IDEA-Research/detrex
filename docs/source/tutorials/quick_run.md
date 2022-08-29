# Quick Run
This is a step-by-step tutorial on how to get started with LiBai:
- [Quick Run](#quick-run)
  - [Train Bert-large Model Parallelly](#train-bert-large-model-parallelly)
    - [Prepare the Data and the Vocab](#prepare-the-data-and-the-vocab)
    - [How to Train Bert_large Model with Parallelism](#how-to-train-bert_large-model-with-parallelism)
  - [Train VisionTransformer on ImageNet Dataset](#train-visiontransformer-on-imagenet-dataset)
    - [Prepare the Data](#prepare-the-data)
    - [Train vit Model from Scratch](#train-vit-model-from-scratch)


## Train Bert-large Model Parallelly
### Prepare the Data and the Vocab

- We have prepared relevant datasets, which can be downloaded from the following links:

1. [VOCAB_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt)
2. [BIN_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin)
3. [IDX_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx)

- Download the dataset and move the data file to the folder. The file structure should be like:
```bash
$ tree data
path/to/bert_data
├── bert-base-chinese-vocab.txt
└── data
    ├── loss_compara_content_sentence.bin
    └── loss_compara_content_sentence.idx
```
### How to Train Bert_large Model with Parallelism

We provide `train.sh` for execute training. Before invoking the script, perform the following steps.

**Step 1. Set data path and vocab path**

- Update the data path and vocab path in [bert_large_pretrain](https://github.com/Oneflow-Inc/libai/blob/main/configs/bert_large_pretrain.py) config file:
```python
# Refine data path and vocab path to data folder
vocab_file = "/path/to/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "/path/to/bert_data/data/loss_compara_content_sentence"
```

**Step 2. Configure your parameters**
- In the [`configs/bert_large_pretrain.py`](https://github.com/Oneflow-Inc/libai/blob/main/configs/bert_large_pretrain.py) provided, a set of parameters are defined including training scheme, model, etc.
- You can also modify the parameters setting. For example, if you want to use 8 GPUs for training, you can refer to the file [`configs/common/train.py`](https://github.com/Oneflow-Inc/libai/blob/main/configs/common/train.py). If you want to train model with 2D mesh hybrid parallelism (4 groups for data parallel and 2 groups for tensor parallel), you can set the the parameters as follows:

```python
train.dist.data_parallel_size=4
train.dist.tensor_parallel_size=2
```

**Step 3. Invoke parallel training**
- To train `BertForPreTraining` model on a single node with 8 GPUs, run:
```bash
bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py 8
```

- To train `BertForPreTraining` model on 2 nodes with 16 GPUs, 
  
  in `node0`, run:
  ```bash
  NODE=2 NODE_RANK=0 ADDR=192.168.0.0 PORT=12345 bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py 8
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=0` means current node is node0

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0

  in `node1`, run:
  ```bash
  NODE=2 NODE_RANK=1 ADDR=192.168.0.0 PORT=12345 bash tools/train.sh tools/train_net.py configs/bert_large_pretrain.py 8
  ``` 
  `NODE=2` means total number of nodes
  
  `NODE_RANK=1` means current node is node1

  `ADDR=192.168.0.0` means the ip address of node0

  `PORT=12345` means the port of node0

## Train VisionTransformer on ImageNet Dataset
### Prepare the Data
For ImageNet, we use standard ImageNet dataset, which can be downloaded from http://image-net.org/.
- For the standard folder dataset, move validation images to labeled sub-folders. The file structure should be like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...

```
### Train vit Model from Scratch
- Update the data path in [vit_imagenet](https://github.com/Oneflow-Inc/libai/blob/main/configs/vit_imagenet.py) config file:
```python
# Refine data path to imagenet data folder
dataloader.train.dataset[0].root = "/path/to/imagenet"
dataloader.test[0].dataset.root = "/path/to/imagenet"
```
- To train `vit_tiny_patch16_224` model on ImageNet on a single node with 8 GPUs for 300 epochs, run:
```bash
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py 8
```
- The default vit model in LiBai is set to `vit_tiny_patch16_224`. To train other vit models, update the [vit_imagenet](https://github.com/Oneflow-Inc/libai/blob/main/configs/vit_imagenet.py) config file by importing other vit models in the config file as follows:
```python
# from .common.models.vit.vit_tiny_patch16_224 import model
from .common.models.vit.vit_base_patch16_224 import model
```
