from detrex.config import get_config
from .models.deta_r50 import model
from .scheduler.coco_scheduler import lr_multiplier_12ep_8bs_scheduler as lr_multiplier

# using the default optimizer and dataloader
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deta_r50_5scale_12ep_bs8"

# max training iterations
train.max_iter = 180000
train.eval_period = 15000
train.checkpointer.period = 15000


# only freeze stem during training
model.backbone.freeze_at = 1 


# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

