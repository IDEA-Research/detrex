from .dino_focalnet_large_lrf_384_4scale_12ep import (
    model,
    dataloader,
    train,
    lr_multiplier,
    optimizer
)

from detrex.config import get_config

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

# modify training config
train.max_iter = 270000
train.init_checkpoint = "/path/to/focalnet_large_lrf_384.pth"
train.output_dir = "./output/dino_focalnet_large_4scale_36ep"
