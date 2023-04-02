from detrex.config import get_config
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# get default config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_r50_4scale_24ep"

# max training iterations
train.max_iter = 180000

# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 16

