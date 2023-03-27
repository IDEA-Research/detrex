from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import InternImage

from .dino_r50 import model


# internimage-large-4scale baseline
model.backbone = L(InternImage)(
    core_op="DCNv3",
    channels=160,
    depths=[5, 5, 22, 5],
    groups=[10, 20, 40, 80],
    mlp_ratio=4.,
    drop_path_rate=0.4,
    norm_layer="LN",
    layer_scale=1.0,
    offset_scale=2.0,
    post_norm=True,
    with_cp=False,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=320),
    "p2": ShapeSpec(channels=640),
    "p3": ShapeSpec(channels=1280),
}
model.neck.in_features = ["p1", "p2", "p3"]
