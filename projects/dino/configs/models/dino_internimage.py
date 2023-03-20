from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import InternImage

from .dino_r50 import model


# focalnet-large-4scale baseline
model.backbone = L(InternImage)(
    core_op="DCNv3",
    channels=64,
    depths=[4, 4, 18, 4],
    groups=[4, 8, 16, 32],
    mlp_ratio=4.,
    drop_path_rate=0.2,
    norm_layer="LN",
    layer_scale=1.0,
    offset_scale=1.0,
    post_norm=False,
    with_cp=False,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=64),
    "p2": ShapeSpec(channels=128),
    "p3": ShapeSpec(channels=256),
}
model.neck.in_features = ["p1", "p2", "p3"]
