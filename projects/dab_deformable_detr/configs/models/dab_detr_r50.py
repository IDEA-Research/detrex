from ideadet.modeling.utils import Joiner, MaskedBackbone
from ideadet.modeling.matcher import DabMatcher
from ideadet.modeling.criterion import DabCriterion
from ideadet.layers import (
    PositionEmbeddingSine,
)

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

from modeling import (
    DabDeformableDETR,
    DabDeformableDetrTransformer,
    DeformableDetrEncoder,
    DabDeformableDetrTransformerDecoder
)


model = L(DabDeformableDETR)(
    backbone=L(Joiner)(
        backbone=L(MaskedBackbone)(
            backbone=L(ResNet)(
                stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
                stages=L(ResNet.make_default_stages)(
                    depth=50,
                    stride_in_1x1=False,
                    norm="FrozenBN",
                ),
                out_features=["res3", "res4", "res5"],
                freeze_at=2,
            )
        ),
        position_embedding=L(PositionEmbeddingSine)(
            num_pos_feats=128, temperature=20, normalize=True
        ),
    ),
    transformer=L(DabDeformableDetrTransformer)(
        encoder=L(DeformableDetrEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.,
            ffn_dropout=0.,
            num_layers=6,
            post_norm=False,
        ),
        decoder=L(DabDeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.,
            ffn_dropout=0.,
            num_layers=6,
            return_intermediate=True,
            use_dab=True,
        ),
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
    ),
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    num_feature_levels=4,
    criterion=L(DabCriterion)(
        num_classes=80,
        matcher=L(DabMatcher)(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
        ),
        weight_dict={
            "loss_ce": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
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
