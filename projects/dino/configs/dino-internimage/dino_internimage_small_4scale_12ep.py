from detectron2.layers import ShapeSpec

from .dino_internimage_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify model to internimage-small version
model.backbone.channels = 80
model.backbone.depths = [4, 4, 21, 4]
model.backbone.groups = [5, 10, 20, 40]
model.backbone.offset_scale = 1.0
model.backbone.drop_path_rate = 0.1
model.backbone.post_norm = True

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=160),
    "p2": ShapeSpec(channels=320),
    "p3": ShapeSpec(channels=640),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training config
train.init_checkpoint = "/path/to/internimage_s_1k_224.pth"
train.output_dir = "./output/dino_internimage_small_4scale_12ep"