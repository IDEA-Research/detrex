# Getting Started with detrex
This document provides a brief intro of the usage of builtin command-line tools in detrex.

## Inference Demo with Pre-trained Models
We've provided [demo](https://github.com/IDEA-Research/detrex/tree/main/demo) as detectron2 for visualizing the customized input images or videos using pretrained weights.

For visualizing demos:
1. Pick a model and its config file from [projects](https://github.com/IDEA-Research/detrex/tree/main/projects), for example, [dino_swin_large_384_4scale_36ep](https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_384_4scale_36ep.py).
2. Download the pretrained weights from [Model Zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) or the [project's page](https://github.com/IDEA-Research/detrex/tree/main/projects/dino#pretrained-models) (take DINO as an example).
3. Using the provided [demo.py](https://github.com/IDEA-Research/detrex/blob/main/demo/demo.py) to demo the input images or videos. Run it as:

```bash
cd detrex/
python demo/demo.py --config-file projects/dino/configs/dino_swin_large_384_4scale_36ep.py \
                    --input input.jpg \
                    --output visualized_results.jpg \
                    --opts train.init_checkpoint="./dino_swin_large_384_4scale_36ep.pth"

```

To visualize videos:
```bash
cd detrex/
python demo/demo.py --config-file projects/dino/configs/dino_swin_large_384_4scale_36ep.py \
                    --video-input ./demo_video.mp4 \
                    --output visualize_video_results.mp4 \
                    --opts train.init_checkpoint="./dino_swin_large_384_4scale_36ep.pth"
```

For details of the command line arguments, run `python demo/demo.py -h` or look at its source code to understand its behavior.


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
