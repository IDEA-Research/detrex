from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .dino_vitdet_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)


# modify training config
train.max_iter = 375000
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth"
train.output_dir = "./output/dino_vitdet_large_50ep"

# warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[300000, 375000],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)