from ideadet.config import get_config

from .models.detr_r50 import model
from .common.coco_loader import dataloader

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_1x
optimizer = get_config("common/optim.py").SGD
train = get_config("common/train.py").train

train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/weights/converted_new_detr_model.pth"
train.output_dir = "./output"
