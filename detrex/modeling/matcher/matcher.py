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

"""
This is the original implementation of HungarianMatcher which will be deprecated in the next version.

We keep it here because our modified HungarianMatcher module is still under test.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from detrex.layers.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """HungarianMatcher which computes an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_class_type: str = "focal_loss_cost",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_cost",
            "focal_loss_cost",
        }, "only support ce loss or focal loss for computing class cost"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Forward function for `HungarianMatcher` which performs the matching.

        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:

                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.

            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:

                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.

        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:

                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)

            For each batch element, it holds: `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.cost_class_type == "ce_cost":
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
        elif self.cost_class_type == "focal_loss_cost":
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).sigmoid()
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

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

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
            "cost_class_type: {}".format(self.cost_class_type),
            "focal cost alpha: {}".format(self.alpha),
            "focal cost gamma: {}".format(self.gamma),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
