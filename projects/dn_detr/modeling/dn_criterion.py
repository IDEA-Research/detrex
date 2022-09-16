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

from detrex.modeling import BaseCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class DNCriterion(BaseCriterion):
    """This class computes the loss for DN-DETR."""

    def forward(self, outputs, targets):
        losses = super(DNCriterion, self).forward(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()


        dn_losses = self.calculate_denoising_loss(outputs, targets, num_boxes)
        losses.update(dn_losses)

        return losses

    def calculate_denoising_loss(self, outputs, targets, num_boxes):
        """
        Calculate denoising loss for DN-DETR
        """

        losses = {}
        if outputs and "denoising_output" in outputs:
            denoising_output, denoising_groups, max_gt_num_per_image = (
                outputs["denoising_output"],
                outputs["denoising_groups"],
                outputs["max_gt_num_per_image"],
            )

            # Collect preds and targets
            noised_logits = denoising_output["pred_logits"]
            noised_boxes = denoising_output["pred_boxes"]
            device = noised_logits.device   

            # Preprocess targets for different denoising groups in each image
            denoising_indices = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    target_idx_per_group = torch.arange(0, len(targets[i]["labels"])).long().to(device)  # target index in one group
                    target_idx_per_group = target_idx_per_group.unsqueeze(0).repeat(denoising_groups, 1)  # repeat for each group
                    target_idx = target_idx_per_group.flatten()
                    output_idx = (
                        torch.tensor(range(denoising_groups)).to(device) * max_gt_num_per_image
                    ).unsqueeze(1) + target_idx_per_group
                    output_idx = output_idx.flatten()
                else:
                    output_idx = target_idx = torch.tensor([]).long().cuda()
                denoising_indices.append((output_idx, target_idx))
            
            losses["loss_class_dn"] = self.calculate_class_loss(noised_logits, targets, denoising_indices, num_boxes * denoising_groups)
            losses["loss_bbox_dn"] = self.calculate_bbox_loss(noised_boxes, targets, denoising_indices, num_boxes * denoising_groups)
            losses["loss_giou_dn"] = self.calculate_giou_loss(noised_boxes, targets, denoising_indices, num_boxes * denoising_groups)
        else:
            device = outputs["pred_logits"].device
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to(device)
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to(device)
            losses["loss_class_dn"] = torch.as_tensor(0.0).to(device)


        # Compute auxiliary denoising losses
        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])

        for i in range(aux_num):
            if outputs and "denoising_output" in outputs:
                denoising_output_aux = denoising_output["aux_outputs"][i]
                aux_noised_logits = denoising_output_aux["pred_logits"]
                aux_noised_boxes = denoising_output_aux["pred_boxes"]
                losses["loss_class_dn" + f"_{i}"] = self.calculate_class_loss(aux_noised_logits, targets, denoising_indices, num_boxes * denoising_groups)
                losses["loss_bbox_dn" + f"_{i}"] = self.calculate_bbox_loss(aux_noised_boxes, targets, denoising_indices, num_boxes * denoising_groups)
                losses["loss_giou_dn" + f"_{i}"] = self.calculate_giou_loss(aux_noised_boxes, targets, denoising_indices, num_boxes * denoising_groups)
            else:
                losses["loss_class_dn"] = torch.as_tensor(0.0).to(device)
                losses["loss_bbox_dn"] = torch.as_tensor(0.0).to(device)
                losses["loss_giou_dn"] = torch.as_tensor(0.0).to(device)
                losses["loss_class_dn" + f"_{i}"] = torch.as_tensor(0.0).to(device)
                losses["loss_bbox_dn" + f"_{i}"] = torch.as_tensor(0.0).to(device)
                losses["loss_giou_dn" + f"_{i}"] = torch.as_tensor(0.0).to(device)

        return losses