from .dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dn_detr_r50_dc5 import model

# modify training config
train.init_checkpoint = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
train.output_dir = "./output/dab_detr_r50_dc5_50ep"

