from detrex.config import get_config

from .dino_focalnet_large_lrf_384_fl4_5scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

# modify training config
train.max_iter = 270000
train.init_checkpoint = "/path/to/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "./output/dino_focalnet_large_fl4_5scale_36ep"