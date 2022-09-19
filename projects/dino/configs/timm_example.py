from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
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
    model_name="ghostnet_100",  # name in timm
    features_only=True,
    pretrained=True,
    in_channels=3,
    out_indices=(1, 2, 3),
    out_features=("p1", "p2", "p3"),
)

# modify neck configs
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=24),
    "p2": ShapeSpec(channels=40),
    "p3": ShapeSpec(channels=80),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training configs
train.init_checkpoint = ""