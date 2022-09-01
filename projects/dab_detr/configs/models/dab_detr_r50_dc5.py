from detectron2.config import LazyCall as L
from detrex.modeling.backbone import ResNet, BasicStem, make_stage

from .dab_detr_r50 import model


model.backbone = L(ResNet)(
    stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
    stages=L(make_stage)(
        depth=50,
        stride_in_1x1=False,
        norm="FrozenBN",
        res5_dilation=2,
    ),
    out_features=["res2", "res3", "res4", "res5"],
    freeze_at=1,
)
