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


