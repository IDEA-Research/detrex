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


import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class DNDeformableDETR(nn.Module):
    def __init__(
        self,
        backbone,
        position_embedding,
        neck,
        transformer,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        embed_dim=256,
        aux_loss=True,
        as_two_stage=False,
        dn_num=5,
        label_noise_scale=0.2,
        box_noise_scale=0.4,
        device="cuda",
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.num_queries = num_queries
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # leave one dim for indicator
        self.label_encoder = nn.Embedding(num_classes + 1, embed_dim - 1)
        self.dn_num = dn_num
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        if not as_two_stage:
            self.tgt_embed = nn.Embedding(num_queries, embed_dim-1)
            self.refpoint_embed = nn.Embedding(num_queries, 4)

        self.aux_loss = aux_loss
        self.as_two_stage = as_two_stage
        self.criterion = criterion

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1) if as_two_stage else transformer.decoder.num_layers
        )
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])

        # hack implementation for iterative bounding box refinement
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        if self.as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed

        self.init_weights()

    def init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for class_embed_layer in self.class_embed:
            class_embed_layer.bias.data = torch.ones(self.num_classes) * bias_value

        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data, 0)

        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for bbox_embed_layer in self.bbox_embed:
                nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

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

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
        else:
            targets = None

        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        tgt_embed = self.tgt_embed.weight  # nq, 255
        refanchor = self.refpoint_embed.weight  # nq, 4

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
            tgt_embed=tgt_embed,
        )

        query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2).transpose(0, 1).contiguous()
        # indicator_mt = torch.zeros([self.num_queries, 1]).to(self.device)
        # input_query_label = torch.cat([tgt_embed, indicator_mt], dim=1).repeat(batch_size, 1, 1)
        # query_embeds = torch.cat((input_query_label, refanchor.repeat(batch_size, 1, 1)), dim=2)

        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds,
            attn_masks=[attn_mask, None],  # None mask for cross attention
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ###### dn post process
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

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

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

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
        tgt_embed,
    ):
        indicator_mt = torch.zeros([num_queries, 1]).to(self.device).contiguous()
        # use learnable tgt embedding in dab_deformable
        if tgt_embed is not None:
            content_queries_mt = tgt_embed
        else:
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
        indicator_dn = torch.ones([input_label_embed.shape[0], 1]).to(self.device).contiguous()
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
