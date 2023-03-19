from .deta_r50_5scale_12ep import (
    model,
    train,
    dataloader,
    lr_multiplier
)

model.backbone.freeze_at = 1
