from detectron2.layers import ShapeSpec
from .dino_focalnet_large_lrf_384_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)


# modify training config
train.init_checkpoint = "/path/to/focalnet_small_lrf.pth"
train.output_dir = "./output/dino_focal_small_lrf_fl3_4scale_12ep"


# convert to focal-small 3level
model.backbone.embed_dim = 96
model.backbone.depths = (2, 2, 18, 2)
model.backbone.focal_levels = (3, 3, 3, 3)
model.backbone.focal_windows = (3, 3, 3, 3)
model.backbone.drop_path_rate = 0.1
model.backbone.use_conv_embed = False
model.backbone.patch_norm = True

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=192),
    "p2": ShapeSpec(channels=384),
    "p3": ShapeSpec(channels=768),
}
model.neck.in_features = ["p1", "p2", "p3"]
