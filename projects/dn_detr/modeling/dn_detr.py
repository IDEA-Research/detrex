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

        # leave one dim for indicator
        self.label_encoder = nn.Embedding(num_classes + 1, embed_dim - 1)
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

        input_query_label, input_query_bbox, attn_mask, dn_metas = self.generate_dn_queries(
            targets,
            dn_num=self.dn_num,
            label_noise_scale=self.label_noise_scale,
            box_noise_scale=self.box_noise_scale,
            refpoint_embed=self.refpoint_embed,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            label_encoder=self.label_encoder,
            batch_size=len(batched_inputs),
        )

        # hs, reference = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])
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
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            loss_dict = self.criterion(output, targets, dn_metas)
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
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), 300, dim=1)
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

    def generate_dn_queries(
        self,
        targets,
        dn_num,
        label_noise_scale,
        box_noise_scale,
        refpoint_embed,
        num_queries,
        num_classes,
        embed_dim,
        label_encoder,
        batch_size,
    ):
        indicator_mt = torch.zeros([num_queries, 1]).to(self.device)
        content_queries_mt = label_encoder(torch.tensor(num_classes).to(self.device)).repeat(
            num_queries, 1
        )
        content_queries_mt = torch.cat([content_queries_mt, indicator_mt], dim=1)

        if targets is None:
            input_query_label = content_queries_mt.repeat(batch_size, 1, 1).transpose(0, 1)
            input_query_bbox = refpoint_embed.weight.repeat(batch_size, 1, 1).transpose(0, 1)
            return input_query_label, input_query_bbox, None, None
        gt_labels = torch.cat([t["labels"] for t in targets])
        gt_boxes = torch.cat([t["boxes"] for t in targets])
        gt_num = [t["labels"].numel() for t in targets]
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )
        unmask_label = torch.cat([torch.ones_like(t["labels"]) for t in targets])
        gt_indices_for_matching = torch.nonzero(unmask_label)
        gt_indices_for_matching = gt_indices_for_matching.view(-1)
        gt_indices_for_matching = gt_indices_for_matching.repeat(dn_num, 1).view(-1)
        dn_bid = batch_idx.repeat(dn_num, 1).view(-1)
        gt_labels = gt_labels.repeat(dn_num, 1).view(-1)
        gt_bboxs = gt_boxes.repeat(dn_num, 1)
        dn_labels = gt_labels.clone()
        dn_boxes = gt_bboxs.clone()

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(dn_labels.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(
                -1
            )  # usually half of bbox noise
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            dn_labels.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(dn_boxes)
            diff[:, :2] = dn_boxes[:, 2:] / 2
            diff[:, 2:] = dn_boxes[:, 2:]
            dn_boxes += (
                torch.mul((torch.rand_like(dn_boxes) * 2 - 1.0), diff).cuda() * box_noise_scale
            )
            dn_boxes = dn_boxes.clamp(min=0.0, max=1.0)

        m = dn_labels.long().to(self.device)
        input_label_embed = label_encoder(m)
        # add dn part indicator
        indicator_dn = torch.ones([input_label_embed.shape[0], 1]).to(self.device)
        input_label_embed = torch.cat([input_label_embed, indicator_dn], dim=1)
        dn_boxes = inverse_sigmoid(dn_boxes)
        single_padding = int(max(gt_num))
        padding_size = int(single_padding * dn_num)
        padding_for_dn_labels = torch.zeros(padding_size, embed_dim).to(self.device)
        padding_for_dn_boxes = torch.zeros(padding_size, 4).to(self.device)

        input_query_label = torch.cat([padding_for_dn_labels, content_queries_mt], dim=0).repeat(
            batch_size, 1, 1
        )
        input_query_bbox = torch.cat([padding_for_dn_boxes, refpoint_embed.weight], dim=0).repeat(
            batch_size, 1, 1
        )

        # map in order
        dn_indices = torch.tensor([]).to(input_query_bbox.device)
        if len(gt_num):
            dn_indices = torch.cat([torch.tensor(range(num)) for num in gt_num])  # [1,2, 1,2,3]
            dn_indices = torch.cat([dn_indices + single_padding * i for i in range(dn_num)]).long()
        if len(dn_bid):
            input_query_label[(dn_bid.long(), dn_indices)] = input_label_embed
            input_query_bbox[(dn_bid.long(), dn_indices)] = dn_boxes

        tgt_size = padding_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(input_query_bbox.device) < 0
        # match query cannot see the reconstruct
        attn_mask[padding_size:, :padding_size] = True
        # reconstruct cannot see each other
        for i in range(dn_num):
            if i == 0:
                attn_mask[
                    single_padding * i : single_padding * (i + 1),
                    single_padding * (i + 1) : padding_size,
                ] = True
            if i == dn_num - 1:
                attn_mask[
                    single_padding * i : single_padding * (i + 1), : single_padding * i
                ] = True
            else:
                attn_mask[
                    single_padding * i : single_padding * (i + 1),
                    single_padding * (i + 1) : padding_size,
                ] = True
                attn_mask[
                    single_padding * i : single_padding * (i + 1), : single_padding * i
                ] = True
        dn_metas = {
            "dn_num": dn_num,
            "single_padding": single_padding,
        }

        input_query_label = input_query_label.transpose(0, 1)
        input_query_bbox = input_query_bbox.transpose(0, 1)
        return input_query_label, input_query_bbox, attn_mask, dn_metas

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
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
