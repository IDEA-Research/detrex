from detectron2.layers import ShapeSpec

from .dino_internimage_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify model to internimage-small version
model.backbone.channels = 112
model.backbone.depths = [4, 4, 21, 4]
model.backbone.groups = [7, 14, 28, 56]
model.backbone.offset_scale = 1.0
model.backbone.drop_path_rate = 0.1
model.backbone.post_norm = True

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=224),
    "p2": ShapeSpec(channels=448),
    "p3": ShapeSpec(channels=896),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training config
train.init_checkpoint = "/path/to/internimage_b_1k_224.pth"
train.output_dir = "./output/dino_internimage_base_4scale_12ep"