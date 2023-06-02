# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from functools import partial
from detectron2.structures import Instances


# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision



def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisibility: int = 0):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        if size_divisibility > 0:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


def tensor_to_cuda(tensor: torch.Tensor, device):
    return tensor.to(device)


def is_tensor_or_instances(data):
    return isinstance(data, torch.Tensor) or isinstance(data, Instances)


def data_apply(data, check_func, apply_func):
    if isinstance(data, dict):
        for k in data.keys():
            if check_func(data[k]):
                data[k] = apply_func(data[k])
            elif isinstance(data[k], dict) or isinstance(data[k], list):
                data_apply(data[k], check_func, apply_func)
            elif isinstance(data[k], int):
                pass
            else:
                raise ValueError()
    elif isinstance(data, list):
        for i in range(len(data)):
            if check_func(data[i]):
                data[i] = apply_func(data[i])
            elif isinstance(data[i], dict) or isinstance(data[i], list):
                data_apply(data[i], check_func, apply_func)
            elif isinstance(data[i], int):
                pass
            else:
                raise ValueError("invalid type {}".format(type(data[i])))
    elif isinstance(data, int):
        pass
    else:
        raise ValueError("invalid type {}".format(type(data)))
    return data


def data_dict_to_cuda(data_dict, device):
    return data_apply(data_dict, is_tensor_or_instances, partial(tensor_to_cuda, device=device))
