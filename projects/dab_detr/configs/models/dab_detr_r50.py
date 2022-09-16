import torch.nn as nn

from detrex.modeling.matcher import (
    ModifedMatcher,
    FocalLossCost,
    L1Cost,
    GIoUCost
)
from detrex.modeling.losses import (
    FocalLoss,
    L1Loss,
    GIoULoss,
)
from detrex.modeling.criterion import BaseCriterion
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L

from projects.dab_detr.modeling import (
    DABDETR,
    DabDetrTransformer,
    DabDetrTransformerDecoder,
    DabDetrTransformerEncoder,
)


model = L(DABDETR)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    in_features=["res5"],  # only use last level feature in DAB-DETR
    in_channels=2048,
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    ),
    transformer=L(DabDetrTransformer)(
        encoder=L(DabDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
        ),
        decoder=L(DabDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
            modulate_hw_attn=True,
        ),
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=300,
    criterion=L(BaseCriterion)(
        num_classes="${..num_classes}",
        matcher=L(ModifedMatcher)(
            cost_class=L(FocalLossCost)(
                alpha=0.25,
                gamma=2.0,
                weight=2.0,
            ),
            cost_bbox=L(L1Cost)(weight=5.0),
            cost_giou=L(GIoUCost)(weight=2.0),
        ),
        loss_class=L(FocalLoss)(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=L(L1Loss)(loss_weight=5.0),
        loss_giou=L(GIoULoss)(
            eps=1e-6,
            loss_weight=2.0,
        ),
    ),
    aux_loss=True,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    freeze_anchor_box_centers=True,
    select_box_nums_for_evaluation=300,
    device="cuda",
)
