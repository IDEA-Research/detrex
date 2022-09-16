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

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, GenerateDNQueries
from detrex.utils.misc import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class DNDETR(nn.Module):
    """Implement DAB-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR 
    <https://arxiv.org/abs/2201.12329>`_
    
    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization. 
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for the initialized dynamic anchor boxes
            in format ``(x, y, w, h)`` and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        denoising_groups (int): Number of groups for noised ground truths. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        with_indicator (bool): If True, add indicator in denoising queries part and matching queries part. 
            Default: True.
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
        freeze_anchor_box_centers: bool = True,
        select_box_nums_for_evaluation: int = 300,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = True,
        device="cuda",
    ):
        super(DNDETR, self).__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding

        # project the backbone output feature 
        # into the required dim for transformer block
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # generate denoising label/box queries
        self.denoising_generator = GenerateDNQueries(
            num_queries=num_queries,
            num_classes=num_classes+1,
            label_embed_dim=embed_dim,
            denoising_groups=denoising_groups,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            with_indicator=with_indicator,
        )
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale

        # define leanable anchor boxes and transformer module
        self.transformer = transformer
        self.anchor_box_embed = nn.Embedding(num_queries, 4)
        self.num_queries = num_queries

        # whether to freeze the initilized anchor box centers during training
        self.freeze_anchor_box_centers = freeze_anchor_box_centers

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(
            input_dim=embed_dim, 
            hidden_dim=embed_dim, 
            output_dim=4, 
            num_layers=3
        )
        self.num_classes = num_classes

        # predict offsets to update anchor boxes after each decoder layer
        # with shared box embedding head
        # this is a hack implementation which will be modified in the future
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # The total nums of selected boxes for evaluation
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        self.init_weights()

    def init_weights(self):
        """Initialize weights for DN-DETR"""
        if self.freeze_anchor_box_centers:
            self.anchor_box_embed.weight.data[:, :2].uniform_(0, 1)
            self.anchor_box_embed.weight.data[:, :2] = inverse_sigmoid(
                self.anchor_box_embed.weight.data[:, :2]
            )
            self.anchor_box_embed.weight.data[:, :2].requires_grad = False

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances): Image meta informations and ground truth boxes and labels during training.
                    Please refer to https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.
        
        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of 
                    each individual image (disregarding possible padding). See PostProcess for information 
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
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

        # only use last level feature as DAB-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # collect ground truth for denoising generation
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
        else:
            # set to None during inference
            targets = None


        # for vallina dn-detr, label queries in the matching part is encoded as "no object" (the last class)
        # in the label encoder.
        matching_label_query = self.denoising_generator.label_encoder(torch.tensor(self.num_classes).to(self.device)).repeat(
            self.num_queries, 1
        )
        indicator_for_matching_part = torch.zeros([self.num_queries, 1]).to(self.device)
        matching_label_query = torch.cat([matching_label_query, indicator_for_matching_part], 1).repeat(batch_size, 1, 1)
        matching_box_query = self.anchor_box_embed.weight.repeat(batch_size, 1, 1)

        if targets is None:
            input_label_query = matching_label_query.transpose(0, 1)  # (num_queries, bs, embed_dim)
            input_box_query = matching_box_query.transpose(0, 1)  # (num_queries, bs, 4)
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
        else:
            # generate denoising queries and attention masks
            noised_label_queries, noised_box_queries, attn_mask, denoising_groups, max_gt_num_per_image = \
                self.denoising_generator(gt_labels_list, gt_boxes_list)        

            # concate dn queries and matching queries as input
            input_label_query = torch.cat([noised_label_queries, matching_label_query], 1).transpose(0, 1)
            input_box_query = torch.cat([noised_box_queries, matching_box_query], 1).transpose(0, 1)


        hidden_states, reference_boxes = self.transformer(
            features,
            img_masks,
            input_box_query,
            pos_embed,
            target=input_label_query,
            attn_mask=[attn_mask, None],  # None mask for cross attention
        )

        # Calculate output coordinates and classes.
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)

        # denoising post process
        output = {'denoising_groups': torch.tensor(denoising_groups).to(self.device),
                  'max_gt_num_per_image': torch.tensor(max_gt_num_per_image).to(self.device)}
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)

        output.update({"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]})
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            loss_dict = self.criterion(output, targets)
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

    def dn_post_process(self, outputs_class, outputs_coord, output):
        if output and output["max_gt_num_per_image"] > 0:
            padding_size = output["max_gt_num_per_image"] * output["denoising_groups"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            output["denoising_output"] = out
        return outputs_class, outputs_coord

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
        """Inference function for DN-DETR
        
        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), 
            self.select_box_nums_for_evaluation, 
            dim=1,
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
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

