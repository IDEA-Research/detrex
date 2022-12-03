from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dino_focalnet import model

# modify training config
train.init_checkpoint = "/path/to/focalnet_large_lrf_384.pth"
train.output_dir = "./output/dino_focalnet_large_4scale_12ep"
