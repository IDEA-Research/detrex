from ideadet.config import get_config

from ideadet.modeling.utils import Joiner, MaskedBackbone
from ideadet.modeling.matcher import DabMatcher
from ideadet.modeling.criterion import DabCriterion
# from ideadet.layers import (
#     PositionEmbeddingSine,
# )
from ..position_encoding import PositionEmbeddingSine

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L

from ..dab_deformable_detr import DABDeformableDETR
from ..deformable_transformer import DeformableTransformer

from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier_12x as lr_multiplier

num_feature_levels=4

model = L(DABDeformableDETR)(
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
                freeze_at=1,
            )
        ),
        position_embedding=L(PositionEmbeddingSine)(
            num_pos_feats=128, temperature=10000, normalize=True
        ),
    ),
    transformer=L(DeformableTransformer)(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward = 2048,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        use_dab=True
    ),
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    num_feature_levels=num_feature_levels,
    criterion=L(DabCriterion)(
        num_classes=80,
        matcher=L(DabMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
        ),
        weight_dict={
            "loss_ce": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        focal_alpha=0.25,
        losses=["labels", "boxes"],
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    two_stage=False,
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.num_decoder_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict


optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/original_dab_deformable.pth"
train.output_dir = "./output/test"
