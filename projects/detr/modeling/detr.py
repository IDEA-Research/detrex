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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/detr.py
# ------------------------------------------------------------------------------------------------

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.layers.mlp import MLP

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class DETR(nn.Module):
    """Implement DAB-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR 
    <https://arxiv.org/abs/2201.12329>`_
    
    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (nn.Module): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization. 
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        device (str): Training device. Default: "cuda".
    """
    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[str],
        in_channels: int,
        position_embedding: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        aux_loss: bool = True,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        device: str = "cuda",
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        
        # project the backbone output feature 
        # into the required dim for transformer block
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # define learnable object queries and transformer module
        self.transformer = transformer
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(
            input_dim=embed_dim, 
            hidden_dim=embed_dim, 
            output_dim=4, 
            num_layers=3
        )
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # only use last level feature in DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        hidden_states, _ = self.transformer(features, img_masks, self.query_embed.weight, pos_embed)

        outputs_class = self.class_embed(hidden_states)
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, box_pred, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
