from detrex.config import get_config
from .models.maskdino_r50 import model
from .data.ade20k_semantic_seg import dataloader

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

train = get_config("common/train.py").train
# max training iterations
train.max_iter = 160000
# warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[135000,150000],
    ),
    warmup_length=10 / train.max_iter,
    warmup_factor=1.0,
)
model.panoptic_on=False
model.instance_on=False
model.semantic_on=True
model.semantic_ce_loss=True
model.sem_seg_head.transformer_predictor.initialize_box_type="no"
model.sem_seg_head.transformer_predictor.two_stage=False
model.sem_seg_head.transformer_predictor.num_queries=100
model.sem_seg_head.transformer_predictor.semantic_ce_loss=True
model.num_queries=100
model.sem_seg_head.pixel_decoder.transformer_dim_feedforward=1024
model.sem_seg_head.pixel_decoder.total_num_feature_levels=3
model.sem_seg_head.pixel_decoder.feature_order="high2low"

model.sem_seg_head.num_classes=150
optimizer = get_config("common/optim.py").AdamW

# initialize checkpoint to be loaded
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dab_detr_r50_50ep"


# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.01
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"


# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 0.05
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# # modify dataloader config
dataloader.train.num_workers = 16
#
# # please notice that this is total batch size.
# # surpose you're using 4 gpus for training and the batch size for
# # each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

