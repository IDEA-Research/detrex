from .dino_swin_large_384_4scale_36ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

from detectron2.layers import ShapeSpec

# modify model config to generate 4 scale backbone features 
# and 5 scale input features
model.backbone.out_indices = (0, 1, 2, 3)

model.neck.input_shapes = {
    "p0": ShapeSpec(channels=192),
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p0", "p1", "p2", "p3"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5

# modify training config
train.output_dir = "./output/dino_swin_large_384_5scale_36ep"