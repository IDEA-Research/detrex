from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .dino_vitdet_base_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# convert vitdet-base to vitdet-large
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.4
# 5, 11, 17, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth"
train.output_dir = "./output/dino_vitdet_large_12ep"


# use warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[300000, 375000],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)