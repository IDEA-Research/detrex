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


def apply_label_noise(
    labels: torch.Tensor, 
    label_noise_scale: float = 0.2, 
    num_classes: int = 80,
):
    """
    Args:
        labels (nn.Tensor): Classification labels with ``(num_labels, )``.

    Returns:
        nn.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_scale > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_scale).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_lebels)
        return noised_labels
    else:
        return labels


def apply_box_noise(
    boxes: torch.Tensor,
    box_noise_scale: float = 0.4,
):
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, :2] = boxes[:, 2:] / 2
        diff[:, 2:] = boxes[:, 2:]
        boxes += (
            torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
        )
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes


class GenerateNoiseQueries(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        noise_nums_per_group: int = 5,
        label_noise_scale: float = 0.0,
        box_noise_scale: float = 0.0,
    ):
        self.num_classes = num_classes
        self.noise_nums_per_group = noise_nums_per_group
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        
        # leave one dim for indicator mentioned in DN-DETR
        self.label_encoder = nn.Embedding(num_classes + 1, label_embed_dim - 1)


