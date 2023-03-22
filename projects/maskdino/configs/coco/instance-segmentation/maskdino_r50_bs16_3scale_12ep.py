from detrex.config import get_config
from ...models.maskdino_r50 import model
from ...data.coco_instance_seg import dataloader

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

# get the default configs
train = get_config("common/train.py").train
optimizer = get_config("common/optim.py").AdamW

# max training iterations
train.max_iter = 90000

# warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[60000, 80000],
    ),
    warmup_length=10 / train.max_iter,
    warmup_factor=1.0,
)

# modify optimizer config
optimizer.lr = 1e-4
optimizer.weight_decay = 0.05
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1  # set backbone lr to 1e-5

# initialize checkpoint to be loaded
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/maskdino_r50_bs16_3scale_50ep"

# set gradient clipping
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2



