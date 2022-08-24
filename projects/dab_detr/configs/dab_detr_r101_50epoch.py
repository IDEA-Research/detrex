from .dab_detr_r50_50epoch import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# modify training config
train.init_checkpoint = "./pretrained_weights/r101.pkl"
train.output_dir = "./output/dab_r101_50epochs"

# modify model config
model.backbone.stages.depth = 101
