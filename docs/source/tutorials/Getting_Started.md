# Getting Started with detrex
This document provides a brief intro of the usage of builtin command-line tools in detrex.

## Data Preparation
In detrex, we use the builtin coco datasets borrowed from detectron2, which has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Here we provide the tutorials about the preparation for `MSCOCO` datasets. For more usage of the detectron2 builtin datasets, please refer to the official documentation: [Use Builtin Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html).

### Expected dataset structure for [COCO instance](https://cocodataset.org/#download)

The dataset structure for `MSCOCO 2017` datasets should be as follows:
```bash
$DETECTRON2_DATASETS/
  coco/
    annotations/
      instances_{train,val}2017.json
      person_keypoints_{train,val}2017.json
    {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets` relative to the current working directory.


## Training & Evaluation in Command Line

In detrex, we provides `tools/train_net.py` for launching training & evaluation task.

### Training & Evaluation
Here we take `dab-detr` as example. To train `dab-detr` using `train_net.py`, first setup the corresponding `MSCOCO 2017` datasets, then run:

```bash
cd detrex
python tools/train_net.py \
    --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py
```

To train on 8 GPUs, you can set `--num-gpus 8` as follows:
```bash
cd detrex
python tools/train_net.py \
    --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \
    --num-gpus 8
```

To evaluate the model performance, use
```bash
python tools/train_net.py \
    --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \
    --eval-only train.init_checkpoint=/path/to/checkpoint
```

**Note:** you can directly modify the config in command line like:
```bash
cd detrex
python tools/train_net.py \
    --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \
    --num-gpus 8 train.max_iter=30000
```
which will directly overide the `train.max_iter` in config.


### Resume Training
If the training is interrupted unexpectly, you can set `--resume` in command line which will automatically resume training from `train.output_dir`:

```bash
python tools/train_net.py \
    --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \
    --num-gpus 8 \
    --resume
```
