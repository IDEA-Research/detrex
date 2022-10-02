from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "/home/rentianhe/code/detrex/r101.pkl"
train.output_dir = "./output/dino_r101_4scale_12ep"

# modify model config
model.backbone.stages.depth = 101
