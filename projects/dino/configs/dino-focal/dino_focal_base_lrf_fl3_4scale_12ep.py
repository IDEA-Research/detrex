from detectron2.layers import ShapeSpec
from .dino_focalnet_large_lrf_384_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
from .focalnet import FocalNet
from detectron2.config import LazyCall as L


# modify training config
train.init_checkpoint = "/path/to/focalnet_base_lrf.pth"
train.output_dir = "./output/dino_focal_small_lrf_fl3_4scale_12ep"


# convert to focal-small 3level
# model.backbone.embed_dim = 128
# model.backbone.depths = (2, 2, 18, 2)
# model.backbone.focal_levels = (3, 3, 3, 3)
# model.backbone.focal_windows = (3, 3, 3, 3)
# model.backbone.drop_path_rate = 0.1
# model.backbone.use_conv_embed = False
# model.backbone.patch_norm = True
# model.backbone.use_postln = False

model.backbone = L(FocalNet)(
    embed_dim=128,
    depths=(2, 2, 18, 2),
    focal_levels=(3, 3, 3, 3),
    focal_windows=(3, 3, 3, 3),
    drop_path_rate=0.1,
    use_conv_embed=False,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=512),
    "p3": ShapeSpec(channels=1024),
}
model.neck.in_features = ["p1", "p2", "p3"]
