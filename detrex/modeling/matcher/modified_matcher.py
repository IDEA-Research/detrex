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

from detrex.layers.box_ops import box_cxcywh_to_xyxy
from detrex.modeling.matcher import FocalLossCost, L1Cost, GIoUCost

class HungarianMatcher(nn.Module):
    """HungarianMatcher which computes an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (nn.Module): Cost function for classification.
        cost_bbox (nn.Module): Cost function for regression L1 cost.
        cost_giou (nn.Module): Cost function for regression iou cost.
    """
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
        gt_labels_list,
        gt_bboxes_list,
    ):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits 
                with shape ``(bs, num_queries, num_class)``.
            pred_bboxes (nn.Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1] with shape
                ``(bs, num_queries, 4)``.
            gt_labels_list (list[nn.Tensor]): Ground truth classification labels for each image
                with shape ``(num_gt,)``.
            gt_bboxes (list[nn.Tensor]): Ground truth boxes with normalized coordinates
                (cx, cy, w, h) for each image, which are all in range [0, 1] with shape
                ``(num_queries, 4)``.
        """
        bs, num_queries, _ = pred_logits.size()
        
        # flatten to compute the cost matrices in a batch
        pred_logits = pred_logits.flatten(0, 1)  # [batch_size * num_queries, num_classes]
        pred_bboxes = pred_bboxes.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        gt_labels = torch.cat(gt_labels_list)
        gt_bboxes = torch.cat(gt_bboxes_list)

        # Compute the classification cost.
        cls_cost = self.cost_class(pred_logits, gt_labels)

        # Compute the L1 cost between boxes
        bbox_cost = self.cost_bbox(pred_bboxes, gt_bboxes)

        # Convert the box format to (x1, y1, x2, y2) to 
        # compute giou cost betwen boxes
        pred_bboxes = box_cxcywh_to_xyxy(pred_bboxes)
        gt_bboxes = box_cxcywh_to_xyxy(gt_bboxes)
        giou_cost = self.cost_giou(pred_bboxes, gt_bboxes)

        # Final cost matrix
        C = cls_cost + bbox_cost + giou_cost
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(gt_boxes) for gt_boxes in gt_bboxes_list]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
