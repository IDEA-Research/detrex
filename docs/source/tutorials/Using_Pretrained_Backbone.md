# Using Pretrained Backbones
This document provides a brief intro of the usage of builtin backbones in detrex.

## ResNet Backbone
### Build ResNet Default Backbone
We modified detectron2 default builtin ResNet models to fit the Lazy Config system. Here we introduce how to implement `ResNet` models or modify it in your own config files.

- Build the default `ResNet-50` backbone

```python
# config.py

from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L

backbone=L(ResNet)(
    stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
    stages=L(ResNet.make_default_stages)(
        depth=50,
        stride_in_1x1=False,
        norm="FrozenBN",
    ),
    out_features=["res2", "res3", "res4", "res5"],
    freeze_at=1,
)
```
**Notes:**
- `stem`: The standard ResNet stem with a `conv`, `relu` and `max_pool`, we usually set `norm="FrozenBN"` to use `FrozenBatchNorm2D` layer in backbone.
- `ResNet.make_default_stages`: This is method which builds the regular ResNet intermediate stages. Set `depth={18, 34, 50, 101, 152}` to build `ResNet-depth` models.
- `out_features`: Set `["res2", "res3"]` to return the intermediate features from the second and third stages.
- `freeze_at`: Set `freeze_at=1` to frozen the backbone at the first stage.

### Build the Modified ResNet Models
- Build `ResNet-DC5` models

```python
from detrex.modeling.backbone import ResNet, BasicStem, make_stage

from detectron2.config import LazyCall as L

backbone=L(ResNet)(
    stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
    stages=L(make_stage)(
        depth=50,
        stride_in_1x1=False,
        norm="FrozenBN",
        res5_dilation=2,
    ),
    out_features=["res2", "res3", "res4", "res5"],
    freeze_at=1,
)
```
- Using the modified `make_stage` function and set `res5_dilation=2` to build `ResNet-DC5` models.
- More details can be found in `make_stage` function [API documentation]()


## Timm Backbone
detrex provides a wrapper for [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models) to use its pretrained backbone networks. Support you want to use `GhostNet` as the backbone of `DAB-Deformable-DETR`, you can modify your config as following:

```python

```
