from detectron2.layers import ShapeSpec

from .dino_convnext_large_384_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify model to tiny version
model.backbone.depths = [3, 3, 9, 3]
model.backbone.dims = [96, 192, 384, 768]

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=192),
    "p2": ShapeSpec(channels=384),
    "p3": ShapeSpec(channels=768),
}
model.neck.in_features = ["p1", "p2", "p3"]

# modify training config
train.init_checkpoint = "/path/to/convnext_tiny_22k_1k_384.pth"
train.output_dir = "./output/dino_convnext_tiny_384_4scale_12ep"
