'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-24 13:53:39
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-30 19:02:41
FilePath: /detrex/projects/co_mot/modeling/mot.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# coding=utf-8

import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from copy import deepcopy
from collections import defaultdict

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from detrex.utils import inverse_sigmoid
from detrex.modeling import SetCriterion
from detrex.modeling.criterion.criterion import sigmoid_focal_loss

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.structures.boxes import matched_pairwise_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized

from projects.co_mot.util import checkpoint
from projects.co_mot.util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy)


class MOT(nn.Module):
    """ Implement CO-MOT: Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking
    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        track_embed: nn.Module,
        track_base: nn.Module,
        post_process: nn.Module,
        aux_loss: bool = True,
        device="cuda",
        g_size=1,
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        self.device = device

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        
        # for Track
        self.track_embed = track_embed
        self.post_process = post_process  # TrackerPostProcess(g_size=g_size)
        self.track_base = track_base
        
        # for shadow
        self.g_size = g_size

        # for init of query
        self.position = nn.Embedding(num_queries, 4)
        self.position_offset = nn.Embedding(num_queries*g_size, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_embed_offset = nn.Embedding(num_queries*g_size, embed_dim)

        nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.normal_(self.position_offset.weight.data, 0, 10e-6)  # 默认为10e-6
        nn.init.normal_(self.query_embed_offset.weight.data, 0, 10e-6) # 默认为10e-6

    def _generate_empty_tracks(self, g_size=1, batch_size=1):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.position.weight.view(-1, 1, 4).repeat(1, g_size, 1).view(-1, 4) + self.position_offset.weight
        track_instances.query_pos = self.query_embed.weight.view(-1, 1, d_model).repeat(1, g_size, 1).view(-1, d_model) +  self.query_embed_offset.weight
        track_instances.ref_pts = track_instances.ref_pts.view(-1, 1, 4).repeat(1, batch_size, 1)
        track_instances.query_pos = track_instances.query_pos.view(-1, 1, d_model).repeat(1, batch_size, 1)

        track_instances.output_embedding = torch.zeros((len(track_instances), batch_size, d_model), device=device)  # motr decode输出的feature，把这个输入qim中可以获得track的query，某个目标跟踪过程中不再使用query_pos
        track_instances.obj_idxes = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device)  # ID
        track_instances.matched_gt_idxes = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device) # 与匹配到的gt在该图片中的索引
        track_instances.disappear_time = torch.zeros((len(track_instances), batch_size), dtype=torch.long, device=device)  # 消失时间，假如目标跟踪多久后删除该目标
        track_instances.iou = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device) #  与对应GT的IOU
        track_instances.scores = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device)   # 实际是当前帧检测输出的置信度
        track_instances.track_scores = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), batch_size, 4), dtype=torch.float, device=device)  # 检测或跟踪query输出的box框
        track_instances.pred_logits = torch.zeros((len(track_instances), batch_size, self.num_classes), dtype=torch.float, device=device)  # 检测或跟踪query输出的置信度argsigmod
        track_instances.group_ids = torch.arange(g_size, dtype=torch.long, device=device).repeat(num_queries).view(-1, 1).repeat(1, batch_size)
        track_instances.labels = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    def _forward_single_image(self, samples, track_instances, gtboxes=None):
        """Forward function of `MOT`.
        """
        
        # original features
        features = self.backbone(samples.tensors)  # output feature dict
        img_masks = samples.mask

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None].float(), size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # prepare label query embedding
        input_query_label = track_instances.query_pos
        input_query_bbox = track_instances.ref_pts
        attn_mask = None

        # feed into transformer 包括encode + decode
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            input_query_label,
            ref_pts=input_query_bbox,
            attn_mask=attn_mask,
        )
        
        # Calculate output coordinates and classes.
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
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output['hs'] = inter_states[-1]
        return output

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores.transpose(0, 1)
        track_instances.pred_logits = frame_res['pred_logits'].transpose(0, 1)
        track_instances.pred_boxes = frame_res['pred_boxes'].transpose(0, 1)
        track_instances.output_embedding = frame_res['hs'].transpose(0, 1)

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)  # 找匹配（跟踪query+检测query）分配GT的ID+ 算loss
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances, g_size=self.g_size)  # 为存在的目标分配ID，并删除长时间消失的目标ID

        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last: # 经过这步后将仅保留有ID的目标，且更新了track的query和pos
            out_track_instances = self.track_embed(tmp, g_size=self.g_size)  # 更新跟踪的query，用于下一帧（检测query为学习的，跟踪query则为上一帧输出经过qim变换的特征）
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None

        # print('post:', t1-t0, t2-t1)
        return frame_res

    # 获取当前帧的跟踪框
    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None): 
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)  # 补pad并或者pad的mask
        if track_instances is None:
            track_instances = self._generate_empty_tracks(g_size=self.g_size) # 初始化decode的输入或者说目标query，包括（query+pose）
        else:
            track_instances = Instances.cat([self._generate_empty_tracks(g_size=self.g_size), track_instances])

        res = self._forward_single_image(img, track_instances=track_instances)  # backbone+encode+decode，获得decode的输出和中间迭代过程输出的box和最后输出的feat(hs,作为经过QIM后可作为下一帧的query)
        res = self._post_process_single_image(res, track_instances, False)  # train是算loss，test时过滤有效跟踪框+为下一帧更新query/pos

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)  # 把box框换算到图像大小
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res: # 把参考点也换算到图像大小
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        # 准备
        def fn(frame, gtboxes, track_instances):
            frame = nested_tensor_from_tensor_list(frame)
            frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            return frame_res

        track_instances = None
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
            frames = data['imgs']  # list of Tensor.
            outputs = {
                'pred_logits': [],
                'pred_boxes': [],
            }
            for frame_index, (frame, gt) in enumerate(zip(frames, data['gt_instances'])):
                for f in frame:
                    f.requires_grad = False
                is_last = frame_index == len(frames) - 1
                nbatch = len(frame)
                gtboxes = None

                if track_instances is None:
                    track_instances = self._generate_empty_tracks(g_size=self.g_size, batch_size=nbatch)
                else:
                    track_instances = Instances.cat([
                        self._generate_empty_tracks(g_size=self.g_size, batch_size=nbatch),
                        track_instances])

                if frame_index < len(frames) - 1:
                    args = [frame, gtboxes, track_instances]
                    params = tuple((p for p in self.parameters() if p.requires_grad))
                    frame_res = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                else:
                    frame = nested_tensor_from_tensor_list(frame)
                    frame_res = self._forward_single_image(frame, track_instances, gtboxes)

                frame_res = self._post_process_single_image(frame_res, track_instances, is_last)
                
                track_instances = frame_res['track_instances']
                outputs['pred_logits'].append(frame_res['pred_logits'])
                outputs['pred_boxes'].append(frame_res['pred_boxes'])
            
            # compute loss
            outputs['losses_dict'] = self.criterion.losses_dict
            loss_dict = self.criterion(outputs, data)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            assert len(data) == 1
            device = self.device
            outputs = []
            for i, data_ in enumerate(data[0]['data_loader']):   # tqdm(loader)):
                cur_img, ori_img, proposals, f_path = [d[0] for d in data_]
                cur_img = cur_img.to(device)
                if track_instances is not None:
                    track_instances.remove('boxes')
                    # track_instances.remove('labels')
                seq_h, seq_w, _ = ori_img.shape

                # 内部包含backboe+encode+decode+跟踪匹配关系+跟踪目标过滤（从query中过滤）
                try: 
                    res = self.inference_single_image(cur_img, (seq_h, seq_w), track_instances)
                except:
                    res = self.inference_single_image(cur_img, (seq_h, seq_w), track_instances)
                track_instances = res['track_instances']

                predictions = deepcopy(res)
                if len(predictions['track_instances']):
                    scores = predictions['track_instances'].scores.reshape(-1, self.g_size)
                    keep_idxs = torch.arange(len(predictions['track_instances']), device=scores.device).reshape(-1, self.g_size)
                    keep_idxs = keep_idxs.gather(1, scores.max(-1)[1].reshape(-1, 1)).reshape(-1)
                    predictions['track_instances'] = predictions['track_instances'][keep_idxs]

                predictions = _filter_predictions_with_confidence(predictions, 0.5)
                predictions = _filter_predictions_with_area(predictions)
                outputs.append(predictions['track_instances'].to('cpu'))
            
            return [outputs]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

def _filter_predictions_with_area(predictions, area_threshold=100):
    if "track_instances" in predictions:
        preds = predictions["track_instances"]
        wh = preds.boxes[:, 2:4] - preds.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep_idxs = areas > area_threshold
        predictions = copy(predictions) # don't modify the original
        predictions["track_instances"] = preds[keep_idxs]
    return predictions
    
def _filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "track_instances" in predictions:
        preds = predictions["track_instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions) # don't modify the original
        predictions["track_instances"] = preds[keep_idxs]
    return predictions


class ClipMatcher(SetCriterion):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        g_size=1
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = defaultdict(float)
        self._current_frame_idx = 0
        self.g_size = g_size

    def initialize_for_single_clip(self, gt_instances: List[Instances]):  # 训练过程中每个视频段之前调用，传入GT值
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = defaultdict(float)

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        gt_instances_i = self.gt_instances[self._current_frame_idx]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances_i],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(self._current_frame_idx, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J) * self.num_classes
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            # loss_class = F.cross_entropy(
            #     src_logits.transpose(1, 2), target_classes, self.empty_weight
            # )
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_class = sigmoid_focal_loss(
                    src_logits.flatten(1),
                    gt_labels_target.flatten(1),
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma
                )

        losses = {"loss_ce": loss_class}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses


    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes[mask]),
                box_cxcywh_to_xyxy(target_boxes[mask])
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def match_for_single_frame(self, outputs: dict):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        
        if not (track_instances.obj_idxes !=-1).any():  # 没有跟踪
            outputs_i = {
                'pred_logits': pred_logits_i.transpose(0,1),
                'pred_boxes': pred_boxes_i.transpose(0,1),
            }
                    
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_i, gt_instances_i, g_size=self.g_size)
            indices = [(ind[0].to(pred_logits_i.device), ind[1].to(pred_logits_i.device)) for ind in indices]


            track_instances.matched_gt_idxes[...] = -1
            for i, ind in enumerate(indices):
                track_instances.matched_gt_idxes[ind[0], i] = ind[1]
                track_instances.obj_idxes[ind[0], i] = gt_instances_i[i].obj_ids[ind[1]].long()
                
                active_idxes = (track_instances.obj_idxes[:, i] >= 0) & (track_instances.matched_gt_idxes[:, i] >= 0) # 当前帧能够匹配到的目标
                active_track_boxes = track_instances.pred_boxes[active_idxes, i]
                if len(active_track_boxes) > 0:
                    gt_boxes = gt_instances_i[i].boxes[track_instances.matched_gt_idxes[active_idxes, i]]
                    active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
                    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                    track_instances.iou[active_idxes, i] = matched_pairwise_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

            self.num_samples += sum(len(t.boxes) for t in gt_instances_i)*self.g_size
            self.sample_device = pred_logits_i.device
            for loss in self.losses:
                new_track_loss = self.get_loss(loss,
                                            outputs=outputs_i,
                                            gt_instances=gt_instances_i,
                                            indices=indices,
                                            num_boxes=1)
                self.losses_dict.update(
                    {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

            if 'aux_outputs' in outputs: # 此处匹配时对新生儿时重新算对应关系的，不直接使用最后一层box输出的对应关系
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    unmatched_outputs_layer = {
                        'pred_logits': aux_outputs['pred_logits'],
                        'pred_boxes': aux_outputs['pred_boxes'],
                    }
                    
                    matched_indices_layer = self.matcher(unmatched_outputs_layer, gt_instances_i, g_size=self.g_size)
                    matched_indices_layer = [(ind[0].to(pred_logits_i.device), ind[1].to(pred_logits_i.device)) for ind in matched_indices_layer]
                    
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        l_dict = self.get_loss(loss,
                                            aux_outputs,
                                            gt_instances=gt_instances_i,
                                            indices=matched_indices_layer,
                                            num_boxes=1, )
                        self.losses_dict.update(
                            {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                            l_dict.items()})
        else:
            track_instances.matched_gt_idxes[...] = -1
            def match_for_single_decoder_layer(unmatched_outputs, matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes):
                new_track_indices = matcher(unmatched_outputs,
                                                [untracked_gt_instances], g_size=self.g_size)  # list[tuple(src_idx, tgt_idx)]

                src_idx = new_track_indices[0][0]
                tgt_idx = new_track_indices[0][1]
                # concat src and tgt.
                new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                                dim=1).to(pred_logits_i.device)
                return new_matched_indices
            for ibn, gt_ins in enumerate(gt_instances_i):
                # step1. inherit and update the previous tracks.
                obj_idxes = gt_ins.obj_ids
                i, j = torch.where(track_instances.obj_idxes[:, ibn:ibn+1] == obj_idxes)  # 获取跟踪query与之相同ID的对应索引
                track_instances.matched_gt_idxes[i, ibn] = j

                full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
                matched_track_idxes = (track_instances.obj_idxes[:, ibn] >= 0)  # occu >=0表明该query为跟踪query
                prev_matched_indices = torch.stack(
                    [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes, ibn]], dim=1)  # 检测或跟踪与gt的对应关系

                # step2. select the unmatched slots.
                # note that the FP tracks whose obj_idxes are -2 will not be selected here.
                unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes[:, ibn] == -1]  # 获取检测query

                # step3. select the untracked gt instances (new tracks).
                tgt_indexes = track_instances.matched_gt_idxes[:, ibn]
                tgt_indexes = tgt_indexes[tgt_indexes != -1]  # 获取跟踪query匹配GT，非新生儿（除了这些之外便是新生儿）

                tgt_state = torch.zeros(len(gt_ins), device=pred_logits_i.device)
                tgt_state[tgt_indexes] = 1 # 新生儿为0，跟踪对应的GT为1
                full_tgt_idxes = torch.arange(len(gt_ins), device=pred_logits_i.device)
                untracked_tgt_indexes = full_tgt_idxes[tgt_state == 0]
                untracked_gt_instances = gt_ins[untracked_tgt_indexes]  # 新生儿的索引

                # step4. do matching between the unmatched slots and GTs.该过程就是DET匈牙利匹配过程
                unmatched_outputs = {
                    'pred_logits': track_instances.pred_logits[unmatched_track_idxes, ibn].unsqueeze(0),
                    'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes, ibn].unsqueeze(0),
                }
                # new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes)
                new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes)

                # step5. update obj_idxes according to the new matching result. 分配GT的ID给track和GT所在的索引
                track_instances.obj_idxes[new_matched_indices[:, 0], ibn] = gt_ins.obj_ids[new_matched_indices[:, 1]].long()
                track_instances.matched_gt_idxes[new_matched_indices[:, 0], ibn] = new_matched_indices[:, 1]

                # step6. calculate iou.
                active_idxes = (track_instances.obj_idxes[:, ibn] >= 0) & (track_instances.matched_gt_idxes[:, ibn] >= 0) # 当前帧能够匹配到的目标
                active_track_boxes = track_instances.pred_boxes[active_idxes, ibn]
                if len(active_track_boxes) > 0:
                    gt_boxes = gt_ins.boxes[track_instances.matched_gt_idxes[active_idxes, ibn]]
                    active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
                    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                    track_instances.iou[active_idxes, ibn] = matched_pairwise_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

                # step7. merge the unmatched pairs and the matched pairs.
                matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

                # step8. calculate losses.
                self.num_samples += len(gt_ins)*self.g_size
                self.sample_device = pred_logits_i.device
                outputs_i = {
                    'pred_logits': pred_logits_i[:, ibn].unsqueeze(0),
                    'pred_boxes': pred_boxes_i[:, ibn].unsqueeze(0),
                }
                for loss in self.losses:
                    new_track_loss = self.get_loss(loss,
                                                    outputs=outputs_i,
                                                    gt_instances=[gt_ins],
                                                    indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                                    num_boxes=1)
                    for key, value in new_track_loss.items():
                        self.losses_dict['frame_{}_{}'.format(self._current_frame_idx, key)] += value

                if 'aux_outputs' in outputs: # 此处匹配时对新生儿时重新算对应关系的，不直接使用最后一层box输出的对应关系
                    for i, aux_outputs in enumerate(outputs['aux_outputs']):
                        unmatched_outputs_layer = {
                            'pred_logits': aux_outputs['pred_logits'][ibn, unmatched_track_idxes].unsqueeze(0),
                            'pred_boxes': aux_outputs['pred_boxes'][ibn, unmatched_track_idxes].unsqueeze(0),
                        }
                        new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, gt_ins[full_tgt_idxes], unmatched_track_idxes, full_tgt_idxes)                        
                        matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                        outputs_layer = {
                            'pred_logits': aux_outputs['pred_logits'][ibn].unsqueeze(0),
                            'pred_boxes': aux_outputs['pred_boxes'][ibn].unsqueeze(0),
                        }
                        for loss in self.losses:
                            if loss == 'masks':
                                # Intermediate masks losses are too costly to compute, we ignore them.
                                continue
                            l_dict = self.get_loss(loss,
                                                outputs_layer,
                                                gt_instances=[gt_ins],
                                                indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                                num_boxes=1, )
                            for key, value in l_dict.items():
                                self.losses_dict['frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key)] += value

        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):  # 实际为一个跟踪ID分配器
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, g_size=1):
        assert track_instances.obj_idxes.shape[1] == 1
        
        device = track_instances.obj_idxes.device

        num_queries = len(track_instances)
        Cindx = torch.arange(num_queries, device=device).reshape(num_queries//g_size, g_size)
        active_idxes = torch.full((num_queries,), False, dtype=torch.bool, device=device)
        # active_idxes[Cindx[track_instances.scores.reshape(-1, g_size).max(-1)[0] >= self.score_thresh].view(-1)] = True
        active_idxes[Cindx[track_instances.scores.reshape(-1, g_size).min(-1)[0] >= self.score_thresh].view(-1)] = True
        # active_idxes[Cindx[track_instances.scores.reshape(-1, g_size).mean(-1) >= self.score_thresh].view(-1)] = True
        track_instances.disappear_time[active_idxes] = 0 # 假如当前帧检测到目标，则disappear_time=0
        
        active_debug = track_instances.scores.reshape(-1, g_size) >= self.score_thresh
        if not (active_debug == active_debug[:,0:1]).any():
            print(track_instances.scores)
        
        new_obj = (track_instances.obj_idxes.reshape(-1) == -1) & (active_idxes) # 挑选新生儿，obj_idxes=-1表示为检测query
        disappeared_obj = (track_instances.obj_idxes.reshape(-1) >= 0) & (~active_idxes)  # 跟踪中假如置信度偏低则 disappear_time++
        num_new_objs = new_obj.sum().item() // g_size

        track_instances.obj_idxes[new_obj, 0] = (self.max_obj_id + torch.arange(num_new_objs, device=device)).view(-1, 1).repeat(1, g_size).view(-1)  # 分配ID
        self.max_obj_id += num_new_objs  # max_obj_id为已有多人ID

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time[:, 0] >= self.miss_tolerance) # 假如当前帧检测不到，且消失很长时间，则把ID删掉，
        track_instances.obj_idxes[to_del, 0] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, g_size=1):
        super().__init__()
        
        self.g_size = g_size

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        # scores = out_logits[..., 0].sigmoid()
        prob = out_logits.sigmoid()
        if len(prob):
            num_query, bn, cls_num = prob.shape
            scores, labels = prob.reshape(-1, self.g_size, bn, cls_num).max(1)[0].reshape(-1, 1, bn, cls_num).repeat(1, self.g_size, 1, 1).reshape(-1, bn, cls_num).max(-1)
        else:
            scores = out_logits[..., 0].sigmoid()
            labels = torch.full_like(scores, 0, dtype=torch.long)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        mask = (track_instances.labels[:, 0] == -1) | (track_instances.labels[:, 0] == labels[:, 0])
        track_instances.labels = labels # torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        # if len(track_instances) != len(track_instances[mask]):
        #     print(track_instances)
        track_instances = track_instances[mask]
        
        return track_instances


def img():
    image = np.ascontiguousarray(((img.tensors[0].permute(1,2,0).cpu()*torch.tensor([0.229, 0.224, 0.225])+torch.tensor([0.485, 0.456, 0.406]))*255).numpy().astype(np.uint8))
    img_h, img_w, _ = image.shape
    bboxes = track_instances.ref_pts.cpu().numpy().reshape(-1, 2, 2)
    bboxes[..., 0] *= img_w
    bboxes[..., 1] *= img_h
    bboxes[:, 0] -= bboxes[:, 1]/2
    bboxes[:, 1] += bboxes[:, 0]
    import cv2
    for i in range(68):
        image_copy = image.copy()
        for box in bboxes[5*i:5*(i+1)]:
            cv2.rectangle(image_copy, pt1 = (int(box[0, 0]), int(box[0, 1])), pt2 =(int(box[1, 0]), int(box[1, 1])), color = (0, 0, 255), thickness = 2)
        cv2.imwrite('tmp2/%d.jpg'%i, image_copy)