import torch.nn as nn

from detrex.layers import (
    MultiheadAttention,
    ConditionalSelfAttention,
    ConditionalCrossAttention,
    PositionEmbeddingSine,
    FFN,
    BaseTransformerLayer,
)
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
            transformer_layers=L(BaseTransformerLayer)(
                attn=L(MultiheadAttention)(
                    embed_dim=256,
                    num_heads=8,
                    attn_drop=0.0,
                    batch_first=False,
                ),
                ffn=L(FFN)(
                    embed_dim=256,
                    feedforward_dim=2048,
                    ffn_drop=0.0,
                    activation=L(nn.PReLU)(),
                ),
                norm=L(nn.LayerNorm)(normalized_shape=256),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DNDetrTransformerDecoder)(
            num_layers=6,
            return_intermediate=True,
            query_dim=4,
            modulate_hw_attn=True,
            post_norm=True,
            transformer_layers=L(BaseTransformerLayer)(
                attn=[
                    L(ConditionalSelfAttention)(
                        embed_dim=256,
                        num_heads=8,
                        attn_drop=0.0,
                        batch_first=False,
                    ),
                    L(ConditionalCrossAttention)(
                        embed_dim=256,
                        num_heads=8,
                        attn_drop=0.0,
                        batch_first=False,
                    ),
                ],
                ffn=L(FFN)(
                    embed_dim=256,
                    feedforward_dim=2048,
                    ffn_drop=0.0,
                    activation=L(nn.PReLU)(),
                ),
                norm=L(nn.LayerNorm)(
                    normalized_shape=256,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    ),
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    query_dim=4,
    iter_update=True,
    random_refpoints_xy=False,
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
    dn_num=5,
    label_noise_scale=0.2,
    box_noise_scale=0.4,
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
