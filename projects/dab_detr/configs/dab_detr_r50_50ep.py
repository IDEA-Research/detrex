from detrex.config import get_config

from .models.dab_detr_r50 import model
from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier


optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/test_dab_detr.pth"
train.output_dir = "./test"
train.max_iter = 375000


# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16
