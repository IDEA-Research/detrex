from fvcore.common.param_scheduler import (
    MultiStepParamScheduler,
    StepParamScheduler,
    StepWithFixedGammaParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
    LinearParamScheduler,
    ExponentialParamScheduler,
)

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def multistep_lr_scheduler(
        values=[1.0, 0.1], 
        warmup_steps=0, 
        num_updates=90000, 
        milestones=[82500], 
        warmup_method="linear", 
        warmup_factor=0.001, 
    ):

    # total steps default to num_updates, if None, will use milestones[-1].
    if num_updates is None:
        total_steps = milestones[-1]
    else:
        total_steps = num_updates

    # define multi-step scheduler
    scheduler = L(MultiStepParamScheduler)(
        values=values,
        milestones=milestones,
        num_updates=num_updates,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )


def step_lr_scheduler(
    values, 
    warmup_steps, 
    num_updates, 
    warmup_method="linear", 
    warmup_factor=0.001, 
):
    
    # define step scheduler
    scheduler = L(StepParamScheduler)(
        values=values,
        num_updates=num_updates
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )


def step_lr_scheduler_with_fixed_gamma(
        base_value,
        num_decays,
        gamma,
        num_updates,
        warmup_steps,
        warmup_method="linear",
        warmup_factor=0.001,
):
    
    # define step scheduler with fixed gamma
    scheduler = L(StepWithFixedGammaParamScheduler)(
        base_value=base_value,
        num_decays=num_decays,
        gamma=gamma,
        num_updates=num_updates,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )


def cosine_lr_scheduler(
    start_value,
    end_value,
    num_updates,
    warmup_steps,
    warmup_method="linear",
    warmup_factor=0.001,
):
    
    # define cosine scheduler
    scheduler = L(CosineParamScheduler)(
        start_value=start_value,
        end_value=end_value,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )

def linear_lr_scheduler(
    start_value,
    end_value,
    num_updates,
    warmup_steps,
    warmup_method="linear",
    warmup_factor=0.001,
):
    
    # define linear scheduler
    scheduler = L(LinearParamScheduler)(
        start_value=start_value,
        end_value=end_value,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )

def constant_lr_scheduler(
    value,
    num_updates,
    warmup_steps,
    warmup_method="linear",
    warmup_factor=0.001,
):
    
    # define constant scheduler
    scheduler = L(ConstantParamScheduler)(
        value=value
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )

def exponential_lr_scheduler(
    start_value,
    decay,
    num_updates,
    warmup_steps,
    warmup_method="linear",
    warmup_factor=0.001,
):
    
    # define exponential scheduler
    scheduler = L(ExponentialParamScheduler)(
        start_value=start_value,
        decay=decay,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )
