from .anchor_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.anchor_detr_r50 import model

# modify training config
train.init_checkpoint = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
train.output_dir = "./output/anchor_detr_r101_50ep"

# modify model
model.backbone.name = "resnet101"
