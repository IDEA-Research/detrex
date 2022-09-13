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

from detrex.utils import inverse_sigmoid


def apply_label_noise(
    labels: torch.Tensor, 
    label_noise_scale: float = 0.2, 
    num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_scale (float):
        num_classes (int): 

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
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
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x1, y1, x2, y2)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): 
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, :2] = boxes[:, 2:] / 2
        diff[:, 2:] = boxes[:, 2:]
        boxes += (
            torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
        )
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes


class GenerateDNQueries(nn.Module):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        noise_nums_per_group: int = 5,
        label_noise_scale: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = False,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.noise_nums_per_group = noise_nums_per_group
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator
        
        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.noise_nums_per_group
        tgt_size = max_gt_num_per_image + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.noise_nums_per_group):
            if i == 0:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
            if i == self.noise_nums_per_group - 1:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1), : max_gt_num_per_image * i
                ] = True
            else:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1), : max_gt_num_per_image * i
                ] = True
        return attn_mask

    def forward(
        self,
        gt_labels_list,
        gt_boxes_list,
    ):
        """
        Args:
            gt_boxes_list (list[torch.Tensor]): Ground truth bounding boxes per image 
                with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)``
            gt_labels_list (list[torch.Tensor]): Classification labels per image in shape ``(num_gt, )``.
        """
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        gt_nums_per_image = [x.numel() for x in gt_labels_list]
        gt_labels = gt_labels.repeat(self.noise_nums_per_group, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.noise_nums_per_group, 1).flatten()

        # noised labels and boxes
        noised_labels = apply_label_noise(gt_labels, self.label_noise_scale, self.num_classes)
        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)


        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        if self.with_indicator:
            label_embedding = torch.cat(
                label_embedding, torch.ones([query_num, 1]).to(device)
            )
        
        max_gt_num_per_image = max(gt_nums_per_image)
        
        noised_query_nums = max_gt_num_per_image * self.noise_nums_per_group

        noised_label_queries = torch.zeros(noised_query_nums, self.label_embed_dim).to(device).repeat(batch_size, 1, 1)
        noised_box_queries = torch.zeros(noised_query_nums, 4).to(device).repeat(batch_size, 1, 1)


        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)
        
        # [0, 0, 0, 0, 1, 1, 2, 2]
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, gt_nums_per_image)
        batch_idx_per_group = batch_idx_per_instance.repeat(self.noise_nums_per_group, 1).flatten()

        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat(torch.tensor(range(num))for num in gt_nums_per_image)
            valid_index_per_group = torch.cat(valid_index_per_group + max_gt_num_per_image * i for i in range(self.noise_nums_per_group)).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_instance, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_instance, valid_index_per_group)] = noised_boxes

        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return noised_label_queries, noised_box_queries, attn_mask
