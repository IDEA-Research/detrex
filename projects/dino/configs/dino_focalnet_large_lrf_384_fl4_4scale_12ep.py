from .dino_focalnet_large_lrf_384_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)


# modify training config
train.init_checkpoint = "/path/to/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "./output/dino_focalnet_large_fl4_4scale_12ep"


# convert to 4 focal-level
model.backbone.focal_levels = (4, 4, 4, 4)
model.backbone.focal_windows = (3, 3, 3, 3)
