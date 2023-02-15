from .dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dab_detr_r50_3patterns_50ep"

# using 3 pattern embeddings as in Anchor-DETR
model.transformer.num_patterns = 3