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

from detrex.modeling import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class DNCriterion(SetCriterion):
    """This class computes the loss for DN-DETR."""

    def forward(self, outputs, targets):
        losses = super(DNCriterion, self).forward(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])

        dn_losses = self.calculate_dn_loss(outputs, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def calculate_dn_loss(self, outputs, targets, aux_num, num_boxes):
        """
        Calculate dn loss in criterion
        """
        losses = {}
        if outputs and "denoising_output" in outputs:
            denoising_output, denoising_groups, single_padding = (
                outputs["denoising_output"],
                outputs["denoising_groups"],
                outputs["max_gt_num_per_image"],
            )
            import pdb
            pdb.set_trace()
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(denoising_groups, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(denoising_groups)).cuda() * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss, denoising_output, targets, dn_idx, num_boxes * denoising_groups, **kwargs
                    )
                )

            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if outputs and "denoising_output" in outputs:
                denoising_output_aux = denoising_output["aux_outputs"][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            denoising_output_aux,
                            targets,
                            dn_idx,
                            num_boxes * denoising_groups,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses