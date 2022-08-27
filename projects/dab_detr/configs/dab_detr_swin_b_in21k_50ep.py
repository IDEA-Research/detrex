from ideadet.config import get_config

from .models.dab_detr_swin_base import model
from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier


optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_base_patch4_window7_224_22k.pth"
train.output_dir = "./output/dab_swin_b_in21k_50ep"
train.max_iter = 375000


# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16
