from detrex.config import get_config
from .models.deta_r50 import model
from .scheduler.coco_scheduler import lr_multiplier_12ep_10drop as lr_multiplier

# using the default optimizer and dataloader
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deta_r50_5scale_12ep"

# max training iterations
train.max_iter = 90000
train.eval_period = 7500
train.checkpointer.period = 7500

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

