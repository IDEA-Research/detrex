from detrex.config import get_config
from .models.dino_swin_tiny_224 import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "/path/to/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
train.output_dir = "./output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune"
train.max_iter = 90000
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.seed = 42

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
