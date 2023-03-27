import torch.nn as nn
from functools import partial
from detectron2.config import LazyCall as L
from detrex.config import get_config
from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detrex.modeling.backbone.eva_vit import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .dino_vitdet_base_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep


# modify training config
train.max_iter = 375000
train.init_checkpoint = "/home/rentianhe/code/detrex/eva_l_psz14_336px_21k_to_1k_ft_89p2.pt"
train.output_dir = "/comp_robot/rentianhe/experiments/dino_eva_large_50ep"

# ViT Base Hyper-params
embed_dim, depth, num_heads, dp = 1024, 24, 16, 0.4

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes = (
            list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
            list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) + list(range(32, 35)) + list(range(36, 39))
        ),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        beit_like_qkv_bias=True,
        beit_like_gamma=False,
        freeze_patch_embed=True,
        use_act_checkpoint=True,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)
