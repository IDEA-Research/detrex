import torch.nn as nn

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.criterion import SetCriterion
from detrex.layers import PositionEmbeddingSine

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

from modeling import (
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
            operation_order=("self_attn", "norm", "ffn", "norm"),
            num_layers=6,
            post_norm=False,
            batch_first=False,
        ),
        decoder=L(DabDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            num_layers=6,
            query_dim=4,
            modulate_hw_attn=True,
            post_norm=True,
            return_intermediate=True,
            batch_first=False,
        ),
    ),
    embed_dim=256,
    in_channels=2048,
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    query_dim=4,
    iter_update=True,
    random_refpoints_xy=True,
    criterion=L(SetCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        losses=[
            "class",
            "boxes",
        ],
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
