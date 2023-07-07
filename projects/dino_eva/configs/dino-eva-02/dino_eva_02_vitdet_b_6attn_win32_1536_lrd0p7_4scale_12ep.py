from functools import partial
from detrex.config import get_config
from detrex.modeling.backbone.eva import get_vit_lr_decay_rate

from ..common.coco_loader_lsj_1536 import dataloader
from ..models.dino_eva_02 import model

# get default config
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train


# modify model config
model.backbone.net.img_size = 1536
model.backbone.square_pad = 1536
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 32
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1  


# 1, 3, 5, 7, 9, 11 for global attention
model.backbone.net.window_block_indexes = [0, 2, 4, 6, 8, 10]

# modify training config
train.init_checkpoint = "/path/to/eva02_B_pt_in21k_p14to16.pt"
train.output_dir = "./output/dino_eva_02_vitdet_b_6attn_win32_1536_lrd0p7_4scale_12ep"

# max training iterations
train.max_iter = 90000


# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

