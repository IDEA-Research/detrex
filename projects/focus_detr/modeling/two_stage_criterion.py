#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

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
from detrex.layers import box_cxcywh_to_xyxy
from detrex.modeling.criterion import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from .foreground_supervision import GenTargets,compute_cls_loss

class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        weight_dict['select_loss'] = 1.5
        self.two_stage_binary_cls = two_stage_binary_cls
        strides = [8, 16, 32, 64]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 999999]]
        self.target_layer = GenTargets(strides, limit_range)
    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        if targets is not None:
            batch_size = len(targets)
            temp_labels = []
            gt_boxx = []
            pad_classes_list = []
            pad_classes90_list = []
            pad_boxes_list = []
            labels = []
            for t in targets:
                h, w = t['size']
                boxes = t["boxes"]
                boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
                boxes = box_cxcywh_to_xyxy(boxes)
                temp_labels.append(t['labels'])
                gt_boxx.append(boxes)
            for c in temp_labels:
                c = torch.ones(c.shape, device=c.device)
                labels.append(c)
            max_num = 0
            for i in range(batch_size):
                n = labels[i].shape[0]
                if n > max_num: max_num = n
            for i in range(batch_size):
                pad_boxes_list.append(
                    torch.nn.functional.pad(gt_boxx[i], (0, 0, 0, max_num - gt_boxx[i].shape[0]), value=-1))
                pad_classes_list.append(torch.nn.functional.pad(labels[i], (0, max_num - labels[i].shape[0]), value=-1))
            batch_classes = torch.stack(pad_classes_list)
            batch_boxes = torch.stack(pad_boxes_list)
            class_targets = self.target_layer(outputs['srcs'], batch_boxes, batch_classes)
            t_mask_pos = (class_targets > 0).squeeze(dim=-1)
        backbone_mask_prediction=outputs["temp_backbone_mask_prediction"]
        select_loss = compute_cls_loss(backbone_mask_prediction, class_targets, t_mask_pos).mean()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        t_l_dict = dict()
        t_l_dict['select_loss'] = select_loss
        losses.update(t_l_dict)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
