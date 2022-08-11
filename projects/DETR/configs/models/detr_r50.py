import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

from ideadet.modeling import DETR, MaskedBackbone, Joiner
from ideadet.layers import (
    BaseTransformerLayer,
    FFN,
    MultiheadAttention,
)
from ideadet.modeling.matcher.matcher import HungarianMatcher
from ideadet.modeling.criterion.criterion import SetCriterion
from ideadet.layers.position_embedding import PositionEmbeddingSine

from modeling import (
    DetrTransformer,
    DetrTransformerEncoder,
    DetrTransformerDecoder,
)

model = L(DETR)(
    backbone=L(Joiner)(
        backbone=L(MaskedBackbone)(
            backbone=L(ResNet)(
                stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
                stages=L(ResNet.make_default_stages)(
                    depth=50,
                    stride_in_1x1=False,
                    norm="FrozenBN",
                ),
                out_features=["res2", "res3", "res4", "res5"],
                freeze_at=2,
            )
        ),
        position_embedding=L(PositionEmbeddingSine)(num_pos_feats=128, normalize=True),
    ),
    transformer=L(DetrTransformer)(
        encoder=L(DetrTransformerEncoder)(
            transformer_layers=L(BaseTransformerLayer)(
                attn=L(MultiheadAttention)(
                    embed_dim=256,
                    num_heads=8,
                    attn_drop=0.1,
                    batch_first=False,
                ),
                ffn=L(FFN)(
                    embed_dim=256,
                    feedforward_dim=2048,
                    ffn_drop=0.1,
                ),
                norm=L(nn.LayerNorm)(normalized_shape=256),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DetrTransformerDecoder)(
            num_layers=6,
            return_intermediate=True,
            transformer_layers=L(BaseTransformerLayer)(
                attn=L(MultiheadAttention)(
                    embed_dim=256,
                    num_heads=8,
                    attn_drop=0.1,
                    batch_first=False,
                ),
                ffn=L(FFN)(
                    embed_dim=256,
                    feedforward_dim=2048,
                    ffn_drop=0.1,
                ),
                norm=L(nn.LayerNorm)(
                    normalized_shape=256,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            post_norm=True,
        ),
    ),
    num_classes=80,
    num_queries=100,
    criterion=L(SetCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
        ),
        weight_dict={
            "loss_ce": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
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
