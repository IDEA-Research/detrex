from .dab_deformable_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify model config
model.as_two_stage = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dab_deformable_detr_r50_two_stage_50ep"
