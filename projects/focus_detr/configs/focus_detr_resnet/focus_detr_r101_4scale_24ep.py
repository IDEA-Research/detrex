from .focus_detr_r50_4scale_24ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "./pre-trained/resnet_torch/r101_v1.pkl"
train.output_dir = "./output/focus_detr_r101_4scale_24ep"

# modify model config
model.backbone.stages.depth = 101
