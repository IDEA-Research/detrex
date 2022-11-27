from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .dino_vitdet_base_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"
train.output_dir = "./output/dino_vitdet_base_100ep"

# max training iterations
train.max_iter = 750000

# modify lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[600000, 750000],
    ),
    warmup_length=0.,
    warmup_method="linear",
    warmup_factor=0.001,
)