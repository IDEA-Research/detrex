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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/models/segmentation.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/dice_loss.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from .utils import weight_reduce_loss


def dice_loss(
    preds,
    targets,
    weight=None,
    eps: float = 1e-4,
    reduction: str = "mean",
    avg_factor: int = None,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor):
            A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-4.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.

    Return:
        torch.Tensor: The computed dice loss.
    """
    preds = preds.flatten(1)
    targets = targets.flatten(1).float()
    numerator = 2 * torch.sum(preds * targets, 1) + eps
    denominator = torch.sum(preds, 1) + torch.sum(targets, 1) + eps
    loss = 1 - (numerator + 1) / (denominator + 1)

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(preds)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class DiceLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=True,
        reduction="mean",
        loss_weight=1.0,
        eps=1e-3,
    ):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
    ):
        if self.use_sigmoid:
            preds = preds.sigmoid()

        loss = self.loss_weight * dice_loss(
            preds,
            targets,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )
        return loss
