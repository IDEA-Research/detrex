from .dino_focalnet_large_lrf_384_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

from detectron2.layers import ShapeSpec


# modify training config
train.init_checkpoint = "/path/to/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "./output/dino_focalnet_large_fl4_5scale_12ep"

# convert backbone to 4 focal-level
model.backbone.focal_levels = (4, 4, 4, 4)
model.backbone.focal_windows = (3, 3, 3, 3)

# convert neck to 5 scale output features
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

# use the original implementation of dab-detr position embedding.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# select top-600 predicted boxes for evaluation
model.select_box_nums_for_evaluation = 600

# using large resolution for testing
dataloader.test.mapper.augmentation[0].short_edge_length = 1200
dataloader.test.mapper.augmentation[0].max_size = 1999