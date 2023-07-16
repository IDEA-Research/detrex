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
# BaseCriterion
# Copyright 2022 The IDEA Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
"""
This is the original implementation of SetCriterion which will be deprecated in the next version.

We keep it here because our modified Criterion module is still under test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from detrex.modeling.criterion import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou,box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from ..losses import binary_cross_entropy_loss_with_logits

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class BaseCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # set some attributes
        self.num_classes = num_classes
        # self.losses = cfg.LOSS.
        # self.pos_norm = #cfg.LO
        self.matcher = matcher
        self.pos_norm_type = 'softmax'
        self.weight_dict = weight_dict

    def _get_src_permutation_idx(self, indices, pos = True):
        
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(v[0], i)
                            for i,v in enumerate(indices)])
        src_idx = torch.cat([v[0] for v in indices])


        return batch_idx, src_idx
   

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(v[1], i)
                              for i,v in enumerate(indices)])
        tgt_idx = torch.cat([v[1] for v in indices])
        return batch_idx, tgt_idx

 

        
    def get_loss(self, outputs:dict, targets:list, num_boxes:float, layer_spec:int, specify_indices=None):
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs
        # assert (num_boxes  >=0)
        assert (layer_spec >=0)
        losses = {}
        pred_boxes = outputs['pred_boxes']
        src_logits = outputs['pred_logits']

        layer_id = layer_spec # not actually used 
        if not specify_indices:
            indices = self.matcher(outputs, targets, 1)#1 to 1 match
        else:
            indices = specify_indices
        # losses.update(info)
        #do preparation work
        target_boxes = torch.cat([t["boxes"][v[1]] for t,v in zip(targets, indices)], dim=0)
        target_classes_o = torch.cat([t["labels"][v[1]]
                                     for t,v in zip(targets, indices)])#1d class index  sorted by tgt_ind
        #compute normalizer for postive
        pos_idx = self._get_src_permutation_idx(indices)
        pos_idx_c = pos_idx + (target_classes_o.cpu(), )
        src_boxes = pred_boxes[pos_idx]
        prob = src_logits.sigmoid().float()        

        # pos_norm,neg_norm = self.get_weight(indices, layer_id, device)
        #extract quality from matcher's output
        # quality = torch.cat([ q['quality'].flatten() for q in indices]).to(device)
        #compute quality here
        alpha = 0.25        
        iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy( target_boxes))[0])
        iou = torch.clamp(iou, 0.01)
        t = prob[pos_idx_c]**alpha * iou ** (1-alpha)
        t = torch.clamp(t, 0.1).detach()

        #compute classification loss
        #define hyper parameters here
        gamma = 2
         
        #define positive weights for SoftBceLoss        
        pos_weights=torch.zeros_like(src_logits)
        pos_weights[pos_idx_c] =  (t -prob[pos_idx_c]) ** gamma
        
        #define negative weights for SoftBceLoss
        neg_weights =  1 * prob ** gamma 
        neg_weights[pos_idx_c] =  (1-t)* prob[pos_idx_c] ** gamma
        
        #sum and average

        loss_ce = binary_cross_entropy_loss_with_logits(src_logits, pos_weights, neg_weights, reduction='mean', avg_factor=num_boxes)    
        losses.update({'loss_class': loss_ce})
        
        #compute regression loss
        loc_weight = 1
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses['loss_bbox'] = (loc_weight* loss_bbox).sum() / num_boxes        
        
        # cmpute giou loss for regression 
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = (loc_weight * loss_giou.view(-1, 1)).sum() / num_boxes
        
        return losses,indices
        
    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = outputs_without_aux['pred_logits'].device
        if 'aux_outputs' in outputs:
            num_layers = len(outputs['aux_outputs']) + 1
        else:
            num_layers = 1
        losses = {}
        indices_list = []
        # match_result = {}
                
        # device = outputs['decoder_coords'][-1].device    
        # layers = len(outputs['decoder_coords'])
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) 
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        
        # final_out = self._prepare_outputs(outputs, -1)        # Compute all the requested losses
        l_dict, indices = self.get_loss( outputs_without_aux, targets,  num_boxes, num_layers)
        losses.update(l_dict)
        indices_list.append(indices)
        
        # indices = None
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if  'aux_outputs' in outputs:
            
            for i in range(num_layers - 1):
                aux_outputs = outputs['aux_outputs'][i]
                l_dict, indices = self.get_loss(aux_outputs, targets, num_boxes , i+1)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                indices_list.append(indices)
                losses.update(l_dict)

        if return_indices:
            return losses, indices
        else:
            return losses
