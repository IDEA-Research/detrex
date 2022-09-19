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
detrex provides a wrapper for [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models) to use its pretrained backbone networks. Support you want to use the pretrained [ResNet-152-D](https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/models/resnet.py#L867) model as the backbone of `DINO`, you can modify your config as following:

```python
from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from detectron2.layers import FrozenBatchNorm2d
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dino_r50 import model

from detrex.modeling.backbone import TimmBackbone

# modify backbone configs
model.backbone = L(TimmBackbone)(
    model_name="resnet152d",  # name in timm
    features_only=True,
    pretrained=True,
    in_channels=3,
    out_indices=(1, 2, 3),
    norm_layer=FrozenBatchNorm2d,
)

# modify neck configs
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=512),
    "p3": ShapeSpec(channels=1024),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training configs
train.init_checkpoint = ""
```
- Set `pretrained=True` which will automatically download pretrained weights from timm.
- Set `features_only=True` to turn timm models into feature extractor.
- Set `out_indices=(1, 2, 3)` which will return the intermediate output feature dict as `{"p1": torch.Tensor, "p2": torch.Tensor, "p3": torch.Tensor}`. 
- Set `norm_layer=nn.Module` to specify the norm layers in backbone, e.g., `norm_layer=FrozenBatchNorm2d` to freeze the norm layers.
- If you want to use timm backbone with your own pretrained weight, please set `pretrained=False` and update `train.init_checkpoint = "path/to/your/own/pretrained_weight/"`

More details can be found in [timm_example.py]()

