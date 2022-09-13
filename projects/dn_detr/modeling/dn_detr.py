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

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils.misc import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class DNDETR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[str],
        transformer: nn.Module,
        position_embedding: nn.Module,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        in_channels: int = 2048,
        embed_dim: int = 256,
        aux_loss: bool = True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=True,
        dn_num=5,
        label_noise_scale=0.0,
        box_noise_scale=0.0,
        device="cuda",
    ):
        super(DNDETR, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.in_features = in_features
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.query_dim = query_dim
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.criterion = criterion

        self.dn_generation = GenerateDNQueries(num_queries,
            num_classes+1,
            label_embed_dim = embed_dim,
            noise_nums_per_group = dn_num,
            label_noise_scale = label_noise_scale,
            box_noise_scale = box_noise_scale,
            with_indicator = True)

        # leave one dim for indicator
        # self.label_encoder = nn.Embedding(num_classes + 1, embed_dim - 1)
        self.dn_num = dn_num
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        assert self.query_dim in [2, 4]

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy

        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.init_weights()

    def init_weights(self):
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):

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

        # only use last level feature in DAB-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        ####### prepare for dn
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
        else:
            targets = None

        gt_labels_list = [t["labels"] for t in targets]
        gt_boxes_list = [t["boxes"] for t in targets]

        # generate dn queries and attn masks
        noised_label_queries, noised_box_queries, attn_mask, noise_nums_per_group, max_gt_num_per_image = \
            self.dn_generation(gt_labels_list, gt_boxes_list,)

        # for vallina dn-detr, label queries in the matching part is encoded as "no object" (the last class)
        # in the label encoder.
        match_query_label = self.dn_generation.label_encoder(torch.tensor(self.num_classes).to(self.device)).repeat(
            self.num_queries, 1
        )
        indicator_mt = torch.zeros([self.num_queries, 1]).to(self.device)
        match_query_label = torch.cat([match_query_label, indicator_mt], 1).repeat(batch_size, 1, 1)
        match_query_bbox = self.refpoint_embed.weight.repeat(batch_size, 1, 1)
        # concate dn queries and matching queries
        input_query_label = torch.cat([noised_label_queries, match_query_label], 1).transpose(0, 1)
        input_query_bbox = torch.cat([noised_box_queries, match_query_bbox], 1).transpose(0, 1)

        hs, reference = self.transformer(
            features,
            img_masks,
            input_query_bbox,
            pos_embed,
            target=input_query_label,
            attn_mask=[attn_mask, None],  # None mask for cross attention
        )

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.bbox_embed(hs)
        tmp[..., : self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()
        outputs_class = self.class_embed(hs)

        ###### dn post process
        output = {'noise_nums_per_group': torch.tensor(noise_nums_per_group).to(self.device),
                  'max_gt_num_per_image': torch.tensor(max_gt_num_per_image).to(self.device)}
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)

        output.update({"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]})
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
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

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

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

    def dn_post_process(self, outputs_class, outputs_coord, output):
        if output and output["max_gt_num_per_image"] > 0:
            padding_size = output["max_gt_num_per_image"] * output["noise_nums_per_group"]
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
        """

        :param max_gt_num_per_image:
        :param device:
        :return:
        """
        noised_query_nums = max_gt_num_per_image * self.noise_nums_per_group
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.noise_nums_per_group):
            if i == 0:
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                max_gt_num_per_image * (i + 1): noised_query_nums,
                ] = True
            if i == self.noise_nums_per_group - 1:
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1), : max_gt_num_per_image * i
                ] = True
            else:
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                max_gt_num_per_image * (i + 1): noised_query_nums,
                ] = True
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1), : max_gt_num_per_image * i
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
                with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, )``
        """
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        gt_nums_per_image = [x.numel() for x in gt_labels_list]
        gt_labels = gt_labels.repeat(self.noise_nums_per_group, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.noise_nums_per_group, 1)

        # noised labels and boxes
        noised_labels = apply_label_noise(gt_labels, self.label_noise_scale, self.num_classes)
        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1]).to(device)], 1)


        max_gt_num_per_image = max(gt_nums_per_image)

        noised_query_nums = max_gt_num_per_image * self.noise_nums_per_group

        noised_label_queries = torch.zeros(noised_query_nums, self.label_embed_dim).to(device).repeat(batch_size, 1, 1)
        noised_box_queries = torch.zeros(noised_query_nums, 4).to(device).repeat(batch_size, 1, 1)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # [0, 0, 0, 0, 1, 1, 2, 2]
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())
        batch_idx_per_group = batch_idx_per_instance.repeat(self.noise_nums_per_group, 1).flatten()

        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.tensor(list(range(num))) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat(
                [valid_index_per_group + max_gt_num_per_image * i for i in range(self.noise_nums_per_group)]).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return noised_label_queries, noised_box_queries, attn_mask, self.noise_nums_per_group, max_gt_num_per_image