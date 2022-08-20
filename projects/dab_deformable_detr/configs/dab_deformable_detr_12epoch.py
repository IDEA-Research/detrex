from ideadet.config import get_config

from .models.dab_detr_r50 import model
from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier_12x as lr_multiplier

optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/test_dab_deformable.pth"
train.output_dir = "./output/test"


optimizer.lr = 1e-4
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 16
