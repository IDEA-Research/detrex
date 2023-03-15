from .deta_r50_5scale_12ep import (
    train,
    model,
    dataloader,
    lr_multiplier,
    optimizer,
)

model.transformer.assign_first_stage = False
model.criterion.assign_first_stage = False
model.criterion.assign_second_stage = False
