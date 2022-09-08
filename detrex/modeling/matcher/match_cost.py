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
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from detrex.layers import (
    generalized_box_iou,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)

class FocalLossCost(nn.Module):
    def __init__(
        self,  
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        weight: float = 1.0, 
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.eps = eps

    def forward(self, pred_logits, gt_labels):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits.
            gt_labels (nn.Tensor): Ground truth labels.

        Return:
            nn.Tensor: Focal loss cost matrix with weight in shape 
                ``(num_queries, num_gt)``
        """
        alpha = self.alpha
        gamma = self.gamma
        eps = self.eps
        out_prob = pred_logits.sigmoid()
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + eps).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + eps).log())
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
        return cost_class * self.weight


class CrossEntropyCost(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight

    def forward(self, pred_logits, gt_labels):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits.
            gt_labels (nn.Tensor): Ground truth labels.

        Return:
            nn.Tensor: CrossEntropy loss cost matrix with weight in shape 
                ``(num_queries, num_gt)``
        """
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        out_prob = pred_logits.softmax(-1)
        cost_class = -out_prob[:, gt_labels]
        return cost_class * self.weight


class GIoUCost(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
    
    def forward(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (nn.Tensor): Predicted bboxes with unnormalized coordinates
                (x1, y1, x2, y2) with shape (num_queries, 4).
            gt_bboxes (nn.Tensor): Ground truth boxes with unnormalized coordinates
                (x1, y1, x2, y2) with shape (num_gt, 4).
        
        Returns:
            torch.Tensor: GIoU cost with weight
        """
        cost_giou = - generalized_box_iou(bboxes, gt_bboxes)
        return cost_giou * self.weight


class L1Cost(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        box_format = "xyxy"
    ):
        super().__init__()
        self.weight = weight
        assert box_format in ["xyxy", "xywh"]
        self.box_format = box_format
    
    def forward(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1] with shape
                (num_queries, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2) with shape (num_gt, 4).

        Returns:
            torch.Tensor: cost_bbox with weight
        """
        if self.box_format == "xywh":
            gt_bboxes = box_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == "xyxy":
            bboxes = box_cxcywh_to_xyxy(bboxes)
        cost_bbox = torch.cdist(bboxes, gt_bboxes, p=1)
        return cost_bbox * self.weight