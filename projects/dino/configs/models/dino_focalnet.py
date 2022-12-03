from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import FocalNet

from .dino_r50 import model


# focalnet-large-4scale baseline
model.backbone = L(FocalNet)(
    embed_dim=192,
    depths=(2, 2, 18, 2),
    focal_levels=(3, 3, 3, 3),
    focal_windows=(5, 5, 5, 5),
    use_conv_embed=True,
    use_postln=True,
    use_postln_in_modulation=False,
    use_layerscale=True,
    normalize_modulator=False,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]
