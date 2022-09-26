from .deformable_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model
)

# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_r50_two_stage_lr_2e4"