from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# no frozen backbone get better results
model.backbone.freeze_at = -1

train.output_dir = "./output/dino_r50_4scale_12ep_no_frozen_backbone"