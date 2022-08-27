from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

from .dab_detr_r50 import model


model.backbone = L(SwinTransformer)(
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.1,
    out_indices=(3,),
)
model.in_features = ["p3"]
model.in_channels = 768
