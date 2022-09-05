from detrex.config import get_config
from .models.deformable_detr_r50 import model

dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_r50_50ep"
train.max_iter = 375000

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "backbone" or "reference_points" or "sampling_offsets" in module_name
    else 1
)

# modify dataloader config
dataloader.train.num_workers = 16
