from .common.models.detr import model
from .common.coco_schedule import lr_multiplier_1x as lr_multiplier
from .common.data.coco_detr import dataloader
from .common.train import train
from .common.optim import SGD as optimizer

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output"
