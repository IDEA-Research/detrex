from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def default_dancetrack_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0, max_iter_epoch=5225):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 8 batch size, using 41796/8=5225
    total_steps_16bs = epochs * max_iter_epoch
    decay_steps = decay_epochs * max_iter_epoch
    warmup_steps = warmup_epochs * max_iter_epoch
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


# default scheduler for detr
lr_multiplier_12ep = default_dancetrack_scheduler(12, 11, 0, 5225)
