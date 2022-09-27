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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from .utils import weight_reduce_loss


def smooth_l1_loss(
    preds,
    targets,
    weight=None,
    beta: float = 1.0,
    reduction: str = "mean",
    avg_factor: int = None,
):
    """Smooth L1 loss.

    Args:
        preds (torch.Tensor): The prediction.
        targets (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if targets.numel() == 0:
        return preds.sum() * 0

    assert preds.size() == targets.size()

    if beta < 1e-5:
        loss = torch.abs(preds - targets)
    else:
        diff = torch.abs(preds - targets)
        cond = diff < beta
        loss = torch.where(cond, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    if weight is not None:
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def l1_loss(
    preds,
    targets,
    weight=None,
    reduction: str = "mean",
    avg_factor: int = None,
):
    if targets.numel() == 0:
        return preds.sum() * 0

    assert preds.size() == targets.size()
    loss = torch.abs(preds - targets)

    if weight is not None:
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class L1Loss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
    ):
        loss_bbox = self.loss_weight * l1_loss(
            preds, targets, weight=weight, reduction=self.reduction, avg_factor=avg_factor
        )
        return loss_bbox
