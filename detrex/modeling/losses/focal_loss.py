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
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/models/segmentation.py
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss

def sigmoid_focal_loss(
    preds, 
    targets, 
    weight = None,
    alpha: float = 0.25, 
    gamma: float = 2,
    reduction: str = "mean",
    avg_factor: int = None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        avg_factor (int): Average factor that is used to average 
            the loss. Default: None.

    Returns:
        torch.Tensor: The computed sigmoid focal loss with the reduction option applied.
    """
    preds = preds.float()
    targets = targets.float()
    p = torch.sigmoid(preds)
    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if weight is not None:
        assert weight.ndim == loss.ndim

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction="mean",
        loss_weight=1.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction=reduction
        self.loss_weight = loss_weight
    
    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
    ):
        loss_class = self.loss_weight * sigmoid_focal_loss(
            preds,
            targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )
        return loss_class