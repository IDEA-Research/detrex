# 使用detectron2的lr_multiplier_3x，学习率，调整学习率衰减
from detrex.config import get_config
from .focus_detr_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# get default config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.init_checkpoint = "./pre-trained/resnet_torch/r50_v1.pkl"
train.output_dir = "./output/focus_detr_r50_4scale_36ep_v3"

# max training iterations
train.max_iter = 270000

# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = True
dataloader.train.num_workers = 16