from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dino_r50 import model

from detrex.modeling.backbone import TorchvisionBackbone

# modify backbone configs
model.backbone = L(TorchvisionBackbone)(
    model_name="resnet50",
    pretrained=True,
    return_nodes = {
        "layer2": "res3",
        "layer3": "res4",
        "layer4": "res5",
    },
)

# modify neck configs
model.neck.input_shapes = {
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res3", "res4", "res5"]

# modify training configs
train.init_checkpoint = ""