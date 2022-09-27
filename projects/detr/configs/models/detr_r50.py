from detectron2.config import LazyCall as L

from detrex.modeling.backbone import ResNet, BasicStem
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.criterion.criterion import SetCriterion
from detrex.layers.position_embedding import PositionEmbeddingSine

from projects.detr.modeling import (
    DETR,
    DetrTransformer,
    DetrTransformerEncoder,
    DetrTransformerDecoder,
)

model = L(DETR)(
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
    in_features=["res5"],
    in_channels=2048,
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
    ),
    transformer=L(DetrTransformer)(
        encoder=L(DetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=2048,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=2048,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            post_norm=True,
        ),
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=100,
    criterion=L(SetCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="ce_cost",
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="ce_loss",
        eos_coef=0.1,
    ),
    aux_loss=True,
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
