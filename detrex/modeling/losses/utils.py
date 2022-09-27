# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/utils.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified

    Args:
        loss (nn.Tensor): Elementwise loss tensor.
        reduction (str): Specified reduction function chosen from "none",
            "mean" and "sum".

    Return:
        nn.Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # Avoid causing ZeroDivisionError when avg_factor is 0.0.
            eps = torch.finfo(loss.dtype).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
