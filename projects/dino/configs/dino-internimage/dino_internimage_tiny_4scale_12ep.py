from detectron2.layers import ShapeSpec

from .dino_internimage_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify model to internimage-tiny version
model.backbone.channels = 64
model.backbone.depths = [4, 4, 18, 4]
model.backbone.groups = [4, 8, 16, 32]
model.backbone.offset_scale = 1.0
model.backbone.drop_path_rate = 0.1
model.backbone.post_norm = False

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=128),
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=512),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training config
train.init_checkpoint = "/path/to/internimage_t_1k_224.pth"
train.output_dir = "./output/dino_internimage_tiny_4scale_12ep"