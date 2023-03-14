from detectron2.config import LazyCall as L
from detrex.modeling.backbone.torchvision_resnet import TorchvisionResNet

from .conditional_detr_r50 import model


model.backbone=L(TorchvisionResNet)(
    name="resnet50",
    train_backbone=True,
    dilation=True,
    return_layers={"layer4": "res5"}
)
