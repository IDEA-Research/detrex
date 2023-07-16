# coding=utf-8

# Copyright 2023 Zhi Cai. All rights reserved.
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
# TwostageCriterion
# Copyright 2022 The IDEA Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
import torch

from .many_to_one_criterion import ManyToOneCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class TwoStageCriterion(ManyToOneCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        match_num,
        gamma,
        alpha,
        tau,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict,match_num,gamma,alpha,tau,
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        
        losses = super().forward(outputs,targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), 1)
        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            # for loss in self.losses:
            l_dict, indices = self.get_loss( enc_outputs, targets, num_boxes, 0)
            l_dict = {k + "_enc": v for k, v in l_dict.items()}
            losses.update(l_dict)

        return losses
