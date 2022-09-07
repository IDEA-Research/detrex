from detrex.config import get_config
from .models.dn_deformable_detr_r50 import model

dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dn_deformable_detr_r50_12ep"
train.max_iter = 90000
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16