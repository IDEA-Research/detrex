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
# Modified from:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/cross_entropy_loss.py
# ------------------------------------------------------------------------------------------------

import warnings
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def cross_entropy(
    preds,
    targets,
    weight=None,
    class_weight=None,
    reduction="mean",
    avg_factor=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    
    loss = F.cross_entropy(
        preds,
        targets,
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = targets.numel() - (targets == ignore_index).sum().item()
    
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )
    
    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=None,
        avg_non_ignore=False,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
        class_weight=None,
        ignore_index=None,
        **kwargs,
    ):
        if ignore_index is None:
            ignore_index = self.ignore_index
        loss_class = self.loss_weight * cross_entropy(
            preds,
            targets,
            weight,
            class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
        )
        return loss_class