from ideadet.config import get_config

from .models.rebuild_dab_detr import model
from .common.coco_loader import dataloader

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_1x
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/converted_dab_r50.pth"
train.output_dir = "./output"
