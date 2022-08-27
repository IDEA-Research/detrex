from .dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from models.dab_detr_swin_base import model

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_base_patch4_window7_224_22k.pth"
train.output_dir = "./output/dab_detr_swin_b_in21k_50ep"
