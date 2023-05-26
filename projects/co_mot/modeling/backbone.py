# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from projects.co_mot.util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from .darknet import CSPDarknet
iter_num = 0
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        out = x * scale + bias
        
        # global iter_num
        # import numpy as np
        # with open('tmp2/bn_%d.txt'%iter_num,'w') as f:
        #     np.savetxt(f,x.view(-1).cpu().detach().numpy()[:1000], delimiter=" ", header='this is begin of x !', footer='this is end of x!', comments='//')
        #     np.savetxt(f,scale.view(-1).cpu().detach().numpy()[:1000], delimiter=" ", header='this is begin of scale !', footer='this is end of scale!', comments='//')
        #     np.savetxt(f,bias.view(-1).cpu().detach().numpy()[:1000], delimiter=" ", header='this is begin of bias !', footer='this is end of bias!', comments='//')
        #     np.savetxt(f,out.view(-1).cpu().detach().numpy()[:1000], delimiter=" ", header='this is begin of out !', footer='this is end of out!', comments='//')
        # print(iter_num)
        # if iter_num == 488:
        #     iter_num=488
        # iter_num+=1
        
        return out


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, name='resnet50'):
        super().__init__()
        if name in ('resnet50'):
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)

            if return_interm_layers:
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                self.num_channels = [2048]
        elif name in ('CSPDarknet'):
            if not train_backbone:
                for name, parameter in backbone.named_parameters():
                    if not train_backbone or 'dark3' not in name and 'dark4' not in name and 'dark5' not in name:
                        parameter.requires_grad_(False)
                backbone.eval()
                
            if return_interm_layers:
                return_layers = {"dark3": "0", "dark4": "1", "dark5": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [320, 640, 1280]
            else:
                return_layers = {'dark5': "0"}
                self.strides = [32]
                self.num_channels = [2048]
                
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

        
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,):
        if name in ('resnet50'):
            norm_layer = FrozenBatchNorm2d
            print(torchvision.__file__, torchvision)
            from .resnet import resnet50
            # backbone = resnet50(
            #     replace_stride_with_dilation=[False, False, dilation],
            #     pretrained=False, norm_layer=norm_layer)  # pretrained=is_main_process()
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=False, norm_layer=norm_layer)  # pretrained=is_main_process()
            assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        elif name in ('CSPDarknet'):
            depth=1.33
            width=1.25
            depthwise=False
            act="silu"
            backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
            def init_yolo(M):
                for m in M.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eps = 1e-3
                        m.momentum = 0.03
            backbone.apply(init_yolo)

        else:
            print("number of channels are hard coded")
        super().__init__(backbone, train_backbone, return_interm_layers, name)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
