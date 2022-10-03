from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer

from .dino_r50 import model


# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=12,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=512),
    "p3": ShapeSpec(channels=1024),
}
model.neck.in_features = ["p1", "p2", "p3"]
