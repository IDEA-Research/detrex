from .pnp_detr_r50_300ep import train, dataloader, optimizer, lr_multiplier, model

# modify model config
model.backbone.stages.depth = 101

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/detr_r101_300ep"
