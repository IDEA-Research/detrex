from detrex.config import get_config
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
)
from .models.dino_r50 import model

# modify lr_multiplier
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_r50_4scale_24ep"
train.max_iter = 180000
