from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

from .dab_detr_r50 import model


model.backbone = L(SwinTransformer)(
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    drop_path_rate=0.4,
    out_indices=(3,),
)
model.in_features = ["p3"]
model.in_channels = 1024
