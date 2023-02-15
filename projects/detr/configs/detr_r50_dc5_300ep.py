from .detr_r50_300ep import dataloader, lr_multiplier, optimizer, train

from .models.detr_r50_dc5 import model

# modify training config
# using torchvision official checkpoint
# the urls can be found in: https://pytorch.org/vision/stable/models/resnet.html

train.init_checkpoint = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
train.output_dir = "./output/detr_r50_dc5_300ep"

