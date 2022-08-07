from ideadet.modeling.meta_arch.detr import DETRDet, Joiner, MaskedBackbone, DETR
from ideadet.layers.transformer import Transformer
from ideadet.modeling.matcher import HungarianMatcher
from ideadet.modeling.criterion import SetCriterion
from ideadet.layers.position_embedding import PositionEmbeddingSine

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

model = L(DETRDet)(
    detr=L(DETR)(
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
        transformer=L(Transformer)(
            d_model=256,
            dropout=0.1,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=6,
            num_decoder_layers=6,
            normalize_before=False,
            return_intermediate_dec=False,
        ),
        num_classes=80,
        num_queries=100,
        aux_loss=True,
    ),
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
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
)
