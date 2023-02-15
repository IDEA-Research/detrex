from .dab_detr_r50_dc5_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# modify training config
train.init_checkpoint = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
train.output_dir = "./output/dab_detr_r50_dc5_3patterns_50ep"

# using 3 pattern embeddings as in Anchor-DETR
model.transformer.num_patterns = 3

# modify model
model.position_embedding.temperature = 20