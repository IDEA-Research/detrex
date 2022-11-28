from detrex.config import get_config

from .dino_vitdet_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# modify lr-multiplier config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# modify training config
train.max_iter = 375000
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth"
train.output_dir = "./output/dino_vitdet_large_50ep"