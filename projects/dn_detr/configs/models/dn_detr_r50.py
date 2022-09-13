import torch.nn as nn

from detrex.layers import PositionEmbeddingSine
from detrex.modeling import HungarianMatcher

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

from projects.dn_detr.modeling import (
    DNDETR,
    DNDetrTransformerEncoder,
    DNDetrTransformerDecoder,
    DNDetrTransformer,
    DNCriterion,
)


model = L(DNDETR)(
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
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    ),
    in_features=["res5"],  # use last level feature as DAB-DETR
    transformer=L(DNDetrTransformer)(
        encoder=L(DNDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DNDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
            modulate_hw_attn=True,
            post_norm=True,
            return_intermediate=True,
        ),
    ),
    num_classes=80,
    num_queries=300,
    in_channels=2048,
    embed_dim=256,
    aux_loss=True,
    freeze_anchor_box_centers=False,
    criterion=L(DNCriterion)(
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
    select_box_nums_for_evaluation=300,
    denoising_groups=5,
    label_noise_prob=0.2,
    box_noise_scale=0.4,
    with_indicator=True,
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    weight_dict["loss_class_dn"] = 1.0
    weight_dict["loss_bbox_dn"] = 5.0
    weight_dict["loss_giou_dn"] = 2.0
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
