from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dino_internimage import model

# modify training config
train.init_checkpoint = "/path/to/internimage_t_1k_224.pth"
train.output_dir = "./output/dino_internimage_t_4scale_12ep"
