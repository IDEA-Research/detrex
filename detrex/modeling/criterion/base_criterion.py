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

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.modeling.matcher import FocalLossCost, L1Cost, GIoUCost, ModifedMatcher
from detrex.modeling.losses import FocalLoss, L1Loss, GIoULoss
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class BaseCriterion(nn.Module):
    """Base criterion for calculating losses for DETR-like models.
    
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes: int,
        matcher = ModifedMatcher(
            cost_class=FocalLossCost(
                alpha=0.25,
                gamma=2.0,
                weight=2.0,
            ),
            cost_bbox=L1Cost(weight=5.0),
            cost_giou=GIoUCost(weight=2.0),
        ),
        loss_class: nn.Module = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
        ),
        loss_bbox: nn.Module = L1Loss(loss_weight=5.0),
        loss_giou: nn.Module = GIoULoss(eps=1e-6, loss_weight=2.0),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_class_loss(self, pred_logits, targets, indices, num_boxes):
        """
        Args:
            preds (torch.Tensor): The predicted logits with shape ``(bs, num_queries, num_classes)``.
            targets (dict): 
            indices (list):
            num_boxes (int):
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Compute classification loss
        pred_logits = pred_logits.view(-1, self.num_classes)
        target_classes = target_classes.flatten()
        losses = self.loss_class(pred_logits, target_classes, avg_factor=num_boxes)
        return losses

    def calculate_bbox_loss(self, pred_boxes, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Compute regression loss
        losses = self.loss_bbox(pred_boxes, target_boxes, avg_factor=num_boxes)
        return losses

    def calculate_giou_loss(self, pred_boxes, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Compute iou loss
        losses = self.loss_giou(pred_boxes, target_boxes, avg_factor=num_boxes)
        return losses

    def forward(self, outputs, targets):
        output_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # Collect preds and targets excluding aux_outputs for matcher
        pred_logits = output_without_aux["pred_logits"]
        pred_boxes = output_without_aux["pred_boxes"]
        target_labels_list = [v["labels"] for v in targets]
        target_boxes_list = [v["boxes"] for v in targets]

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(
            pred_logits,
            pred_boxes,
            target_labels_list,
            target_boxes_list,
        )

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()


        # Compute all losses for DETR-like models
        losses = {}
        losses["loss_class"] = self.calculate_class_loss(pred_logits, targets, indices, num_boxes)
        losses["loss_bbox"] = self.calculate_bbox_loss(pred_boxes, targets, indices, num_boxes)
        losses["loss_giou"] = self.calculate_giou_loss(pred_boxes, targets, indices, num_boxes)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_output in enumerate(outputs["aux_outputs"]):
                aux_pred_logits = aux_output["pred_logits"]
                aux_pred_boxes = aux_output["pred_boxes"]
                indices = self.matcher(aux_pred_logits, aux_pred_boxes, target_labels_list, target_boxes_list)
                losses["loss_class" + f"_{i}"] = self.calculate_class_loss(aux_pred_logits, targets, indices, num_boxes)
                losses["loss_bbox" + f"_{i}"] = self.calculate_bbox_loss(aux_pred_boxes, targets, indices, num_boxes)
                losses["loss_giou" + f"_{i}"] = self.calculate_giou_loss(aux_pred_boxes, targets, indices, num_boxes)
                
        return losses