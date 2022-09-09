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
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/giou_loss.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from .utils import weight_reduce_loss


def giou_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    weight = None,
    eps: float = 1e-6,
    reduction: str = "mean",
    avg_factor: int = None,
):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        preds (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        targets (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if targets.numel() == 0:
        return preds.sum() * 0

    x1, y1, x2, y2 = preds.unbind(dim=-1)
    x1g, y1g, x2g, y2g = targets.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if weight is not None:
        assert weight.ndim == loss.ndim
    
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class GIoULoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
    ):
        loss_giou = self.loss_weight * giou_loss(
            preds,
            targets,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            avg_factor=avg_factor
        )
        return loss_giou
