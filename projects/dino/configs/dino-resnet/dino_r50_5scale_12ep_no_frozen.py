from .dino_r50_5scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

model.backbone.freeze_at = -1

train.output_dir = "./output/dino_r50_5scale_12ep_no_frozen_backbone"