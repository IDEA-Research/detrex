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
# HungarianMatcher
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# https://github.com/alibaba/EasyCV/blob/master/easycv/models/detection/utils/matcher.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from detrex.layers.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.modeling.matcher import FocalLossCost, L1Cost, GIoUCost

class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: nn.Module = FocalLossCost(
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
            eps=1e-8,
        ),
        cost_bbox: nn.Module = L1Cost(weight=5.0),
        cost_giou: nn.Module = GIoUCost(weight=2.0),
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits,
        pred_bboxes,
        gt_labels,
        gt_bboxes,
    ):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits 
                with shape ``(bs, num_queries, num_class)``.
            pred_bboxes (nn.Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1] with shape
                ``(bs, num_queries, 4)``.
            gt_labels (nn.Tensor): Ground truth classification labels with shape
                ``(num_gt,)``.
            gt_bboxes (nn.Tensor): Ground truth boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1] with shape
                ``(num_queries, 4)``.
        """
        bs, num_queries, _ = pred_logits.shape()
        
        # flatten to compute the cost matrices in a batch
        pred_logits = pred_logits.flatten(0, 1)  # [batch_size * num_queries, num_classes]
        pred_bboxes = pred_bboxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if self.cost_class_type == "ce_cost":
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
