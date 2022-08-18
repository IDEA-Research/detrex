from ideadet.config import get_config

from .models.dab_detr_r50 import model
from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier_12x

optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/converted_deformable_dab.pth"
train.output_dir = "./output/r50_base_12ep"

dataloader.train.total_batch_size = 16
