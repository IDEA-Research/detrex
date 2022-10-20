from .h_deformable_detr_r50_50ep import train, dataloader, optimizer, model
from detrex.config import get_config

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/h_deformable_detr_r50_two_stage_12ep"
train.max_iter = 90000
