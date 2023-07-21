#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

from detrex.config import get_config
from .focus_detr_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

# get default config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.init_checkpoint = "./pre-trained/resnet_torch/r50_v1.pkl"
train.output_dir = "./output/focus_detr_r50_4scale_24ep"

# max training iterations
train.max_iter = 180000

# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = True
dataloader.train.num_workers = 16