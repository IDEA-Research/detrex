# Config System

Given that the traditional yacs-based config system or python argparse command-line options suffer from providing enough flexibility for the development of new project, we adopted the alternative and non-intrusive config system called [lazy config](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) from [detectron2](https://github.com/facebookresearch/detectron2).

Please refer to [detectron2 lazy-config tutorials](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more details about the syntax and basic usage of lazy config.


## Default Configs in detrex
Our ``detrex`` has defined a standard set of config namespaces for later usage. Users can modify these configs according to their own needs.

In summary, the pre-defined namespaces are ``model, train, dataloader, optimizer, lr_multiplier``

### model
This is configuration for model definition. We define all model configs under ``projects/``, You can refer to [projects/dab_detr/configs/models](https://github.com/rentainhe/detrex/blob/main/projects/dab_detr/configs/models/dab_detr_r50.py) for more examples.

Here is the example of `dab-detr-r50` model config:

```python
# dab_detr_r50.py
import torch.nn as nn

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.criterion import SetCriterion
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L

from projects.dab_detr.modeling import (
    DABDETR,
    DabDetrTransformer,
    DabDetrTransformerDecoder,
    DabDetrTransformerEncoder,
)


model = L(DABDETR)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    in_features=["res5"],  # only use last level feature in DAB-DETR
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    ),
    transformer=L(DabDetrTransformer)(
        encoder=L(DabDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            operation_order=("self_attn", "norm", "ffn", "norm"),
            num_layers=6,
            post_norm=False,
            batch_first=False,
        ),
        decoder=L(DabDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            num_layers=6,
            query_dim=4,
            modulate_hw_attn=True,
            post_norm=True,
            return_intermediate=True,
            batch_first=False,
        ),
    ),
    embed_dim=256,
    in_channels=2048,
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    query_dim=4,
    iter_update=True,
    random_refpoints_xy=True,
    criterion=L(SetCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        losses=[
            "class",
            "boxes",
        ],
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
)
```
which can be loaded like:
```python
# user's own config.py
from dab_detr_r50 import model

# check the loaded model config
assert model.embed_dim == 256

# modify model config according to your own needs
model.embed_dim = 512
```
After defining model configs in python files. Please ``import`` it in the global scope of the final config file as ``model``. 

You can access and change all keys in the model config according to your own needs.


### train
This is the configuration for training and evalution. The default training config can be found in ``configs/common/train.py``.

The default training config is as follows:
```python
train = dict(

    # Directory where output files are written to
    output_dir="./output",

    # The initialize checkpoint to be loaded
    init_checkpoint="",

    # The total training iterations
    max_iter=90000,

    # options for Automatic Mixed Precision
    amp=dict(enabled=False),

    # options for DistributedDataParallel
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),

    # options for Gradient Clipping during training
    clip_grad=dict(
        enabled=False,
        params=dict(
            max_norm=0.1,
            norm_type=2,
        ),
    ),

    #  # options for Fast Debugging
    fast_dev_run=dict(enabled=False),

    # options for PeriodicCheckpointer, which saves a model checkpoint
    # after every `checkpointer.period` iterations,
    # and only `checkpointer.max_to_keep` number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),

    # Run evaluation after every `eval_period` number of iterations
    eval_period=5000,

    # Output log to console every `log_period` number of iterations.
    log_period=20,
    device="cuda"
    # ...
)
```

### dataloader
This is the configuration for dataset and dataloader. We use the built-in dataset in detectron2 for simplicity. Please see ``configs/common/data`` for more examples.

Here we take the `coco_detr.py` for detr-like model as an example:
```python
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

# the defined train loader
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(DetrDatasetMapper)(
        # the defined two augmentations which will be random-selected during training.
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,  # with instance mask or not
        img_format="RGB",  # input image format
    ),
    total_batch_size=16,  # training batch size
    num_workers=4,
)

# the defined test loader
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        # use no augmentation in testing
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

# the defined evaluator for evaluation
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
```

We adopted the ``built-in coco datasets`` and ``detection dataloader`` usage from detectron2, please refer to the following tutorials if you want to use custom datasets:
- [Use Custom Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)
- [Dataloader](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html)
- [Data Augmentation](https://detectron2.readthedocs.io/en/latest/tutorials/augmentation.html)


### optimizer
This is the configuration for optimizer. The default configuration can be found in ``configs/common/optim.py``.

detrex uilizes ``detectron2.solver.build.get_default_optimizer_params`` which needs the ``nn.Module`` as argument and returns the parameter groups.

```python
# configs/common/optim.py
import torch

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
```
if you want to use ``torch.optim.SGD`` in training, you can modify your config as follows:
```python
import torch
from configs.commom.optim import AdamW as optim

optim._target_ = torch.optim.SGD

# Remove the incompatible arguments
del optim.betas

# Add the needed arguments
optim.momentum = 0.9
```


### lr_multiplier
This is the configuration for ``lr_multiplier`` which is combined with ``detectron2.engine.hooks.LRScheduler`` and  performs learning scheduler function during training.

The default ``lr_multiplier`` config can be found in ``configs/common/coco_schedule.py``, we defined the commonly 50 epochs scheduler referred to in the papers as follows:

```python
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

def default_coco_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 7500
    decay_steps = decay_epochs * 7500
    warmup_steps = warmup_epochs * 7500
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )
```
Please refer to [fvcore.common.param_scheduler.ParamScheduler](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.common.param_scheduler.ParamScheduler) for more details about the ``ParamScheduler`` usage in detectron2.


## Get the Default Config
Users don't have to rewrite all contents in config every time. You can use the default built-in detrex configs using ``detrex.config.get_config``.

After building ``detrex`` from source, you can use ``get_config`` to get the default configs as follows:

```python
from detrex.config import get_config

# get the default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
train = get_config("common/train.py").train

# modify the config
train.max_iter = 375000
train.output_dir = "path/to/your/own/dir"
```

## LazyConfig Best Practices
1. Treat the configs you write as actual "code": Avoid copying them or duplicating them. Import the common parts between configs.
2. Keep the configs you write as simple as possible: Do not include keys that do not affect the experimental setting.
