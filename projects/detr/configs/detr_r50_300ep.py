from detrex.config import get_config
from .models.detr_r50 import model

dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/detr_r50_300ep"
train.max_iter = 554400

# modify lr_multiplier
lr_multiplier.scheduler.milestones = [369600, 554400]

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 64