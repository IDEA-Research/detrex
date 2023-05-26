'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-24 13:53:39
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-26 14:55:15
FilePath: /detrex/projects/co_mot/modeling/co_mot.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# coding=utf-8

import copy
import math
import numpy as np
from copy import deepcopy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# from detrex.utils import inverse_sigmoid

from projects.co_mot.util import box_ops, checkpoint
from projects.co_mot.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .memory_bank import build_memory_bank
from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .qim import build as build_query_interaction_layer
from .deformable_detr import SetCriterion, MLP, sigmoid_focal_loss

from .mot import MOT


class COMOT(MOT):
    """ Implement CO-MOT: Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking
    Args:
        backbone: torch module of the backbone to be used. See backbone.py
        transformer: torch module of the transformer architecture. See transformer.py
        num_classes: number of object classes
        num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                        DETR can detect in a single image. For COCO, we recommend 100 queries.
        aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        with_box_refine: iterative bounding box refinement
        two_stage: two-stage Deformable DETR
    """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        criterion,
        track_embed,
        track_base,
        post_process,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        memory_bank=None,
        use_checkpoint=False,
        query_denoise=0,
        g_size=1,
        args=None,
        device='cuda',
    ):
        super().__init__(backbone,
                        transformer,
                        num_classes,
                        num_queries,
                        num_feature_levels,
                        criterion,
                        track_embed,
                        track_base,
                        post_process,
                        aux_loss,
                        with_box_refine,
                        two_stage,
                        memory_bank,
                        use_checkpoint,
                        query_denoise,
                        g_size,
                        args,
                        device)
        # define backbone
        self.backbone = backbone
        # number of dynamic anchor boxes
        self.num_queries = num_queries
        self.track_embed = track_embed

        # define transformer module
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # define classification head and box head
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_classes = num_classes
        
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.position = nn.Embedding(num_queries, 4)
        self.position_offset = nn.Embedding(num_queries*g_size, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_offset = nn.Embedding(num_queries*g_size, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.normal_(self.position_offset.weight.data, 0, 10e-6)  # 默认为10e-6
        # nn.init.constant_(self.position_offset.weight.data, 0)
        nn.init.normal_(self.query_embed_offset.weight.data, 0, 10e-6) # 默认为10e-6

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = post_process
        self.track_base = track_base
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length
        
        self.g_size = g_size
        self.device = device

    def _generate_empty_tracks(self, proposals=None, g_size=1, batch_size=1):
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

        mem_bank_len = self.mem_bank_len  # 默认是不使用的
        track_instances.mem_bank = torch.zeros((len(track_instances), batch_size, mem_bank_len, d_model), dtype=torch.float32, device=device)  # 存储历史的output_embedding（经过了一个linear）
        track_instances.mem_padding_mask = torch.ones((len(track_instances), batch_size, mem_bank_len), dtype=torch.bool, device=device)  # 是否更新了，假如更新为0，不更新为1
        track_instances.save_period = torch.zeros((len(track_instances), batch_size), dtype=torch.float32, device=device) # 

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        features, pos = self.backbone(samples)  # 通过backbone获取多次度feature+pad mask+position
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):  # 把backbone输出的feature的通道整成一致，方便concat作为encode输入
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):  # 一般backbone仅输出8/16/32的特征，这个添加一个64的特征
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if gtboxes is not None:  # dn-detr方式训练使用，制作一批噪声GT训练decode
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            # attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            # attn_mask[:n_dt, n_dt:] = True
            # attn_mask[n_dt:, :n_dt] = True
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=torch.float32, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = float('-inf')
            attn_mask[n_dt:, :n_dt] = float('-inf')
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        # transformer 包括encode + decode
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):  # 检测头包括box（xc,yc,w,h并归一化1）和class分支
            if lvl == 0:
                reference = init_reference # 实际上就是上一帧跟踪框和decode输入框
            else:
                reference = inter_references[lvl - 1]  # 由于内部是迭代优化的，参考点应为上一次迭代的输出
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
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

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  # 最后一次迭代的输出
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord) # 中间过程迭代的输出
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:  # 由query内有部分噪声GT，因此需要单独拿出来，并放在了ps_outputs下
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

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

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
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
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None): 
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)  # 补pad并或者pad的mask
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals, g_size=self.g_size) # 初始化decode的输入或者说目标query，包括（query+pose）
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals, g_size=self.g_size),
                track_instances])
        # track_instances = self._generate_empty_tracks(proposals)
        res = self._forward_single_image(img,
                                         track_instances=track_instances)  # backbone+encode+decode，获得decode的输出和中间迭代过程输出的box和最后输出的feat(hs,作为经过QIM后可作为下一帧的query)
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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_bdd': 11,
        'e2e_tao': 2000,
        'e2e_bddcc': 100,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
        'e2e_all': 91,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)
    
    g_size = args.g_size

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses, g_size=g_size)
    criterion.to(device)
    postprocessors = {}
    model = MOT(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
        g_size = g_size,
        args=args,
    )
    model.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    return model, criterion, postprocessors


def img():
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
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