# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from pytorch-checkpoint (https://github.com/csrhddlam/pytorch-checkpoint)
# ------------------------------------------------------------------------

import torch


def check_require_grad(t):
    return isinstance(t, torch.Tensor) and t.requires_grad


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            if check_require_grad(temp):
                ctx.input_tensors[i] = temp.detach()
                ctx.input_tensors[i].requires_grad = temp.requires_grad
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        to_autograd = list(filter(check_require_grad, ctx.input_tensors))
        output_tensors, output_grads = zip(*filter(lambda t: t[0].requires_grad, zip(output_tensors, output_grads)))
        input_grads = torch.autograd.grad(output_tensors, to_autograd + ctx.input_params, output_grads, allow_unused=True)
        input_grads = list(input_grads)
        for i in range(len(ctx.input_tensors)):
            if not check_require_grad(ctx.input_tensors[i]):
                input_grads.insert(i, None)
        return (None, None) + tuple(input_grads)
