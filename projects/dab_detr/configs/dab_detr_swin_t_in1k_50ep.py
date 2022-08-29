from .dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dab_detr_swin_tiny import model

# modify training config
train.init_checkpoint = "path/to/swin_tiny_patch4_window7_224.pth"
train.output_dir = "./output/dab_detr_tiny_in1k_50ep"
