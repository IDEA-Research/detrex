from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import ConvNeXt

from .dino_r50 import model


# convnext-large-4scale baseline
model.backbone = L(ConvNeXt)(
    in_chans=3,
    depths=[3, 3, 27, 3],
    dims=[192, 384, 768, 1536],
    drop_path_rate=0.0,
    layer_scale_init_value=1.0,
    out_indices=[1, 2, 3],
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]
