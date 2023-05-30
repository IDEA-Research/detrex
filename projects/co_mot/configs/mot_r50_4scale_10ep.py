'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-25 09:54:44
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-30 15:13:37
FilePath: /detrex/projects/co_mot/configs/mot_r50_4scale_10ep.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from detrex.config import get_config
from .mot_r50 import model

# get default config
dataloader = get_config("common/data/dancetrack_mot.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep  # 这个需要改
# lr_multiplier = 
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/MOTRv2/detrex/output/mot_r50_4scale_12ep"

# dancetrack 41796 imgs
# max training iterations
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 100
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# 
train.lr_backbone_names = ['backbone.0']
train.lr_linear_proj_names = ['reference_points', 'sampling_offsets',]

# for ddp
train.ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=True,
        fp16_compression=False,
    )

# modify optimizer config
optimizer.lr = 2e-4
optimizer.lr_backbone = 2e-5
optimizer.lr_linear_proj_mult = 0.1

optimizer.sgd=False
optimizer.weight_decay = 1e-4

optimizer.betas = (0.9, 0.999)
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
