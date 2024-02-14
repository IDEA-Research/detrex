from detrex.config import get_config
from .deta_r50_5scale_12ep import (
    train,
    optimizer,
)

from .models.deta_swin import model
# from .data.coco_detr_larger import dataloader
from .data.ab_detr_larger_4_cls import dataloader


# 24ep for finetuning
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

# modify learning rate
optimizer.lr = 5e-5
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
