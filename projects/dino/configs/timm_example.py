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