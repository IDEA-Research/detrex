from .dab_detr_r50_50ep import (
    model,
    dataloader,
    lr_multiplier,
    optimizer,
    train
)

# modify training config
train.init_checkpoint = "./pretrained_weights/r101.pkl"
train.output_dir = "./output/dab_detr_r101_50ep"

# modify model config
model.backbone.stages.depth = 101
