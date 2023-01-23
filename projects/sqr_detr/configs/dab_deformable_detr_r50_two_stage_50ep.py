from .dab_deformable_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dab_deformable_detr_r50_two_stage_50ep"

# modify model config
model.as_two_stage = True

# modify loss weight dict
# this is an hack implementation which will be improved in the future
aux_weight_dict = {
    "loss_class_enc": 1.0,
    "loss_bbox_enc": 5.0,
    "loss_giou_enc": 2.0,
}
model.criterion.weight_dict.update(aux_weight_dict)
