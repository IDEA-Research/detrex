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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/mmdet/models/backbones/convnext.py
# ------------------------------------------------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from detrex.layers import LayerNorm

from detectron2.modeling.backbone import Backbone


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(Backbone):
    r"""Implement paper `A ConvNet for the 2020s <https://arxiv.org/pdf/2201.03545.pdf>`_.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (Sequence[int]): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (List[int]): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
    ):
        super().__init__()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        assert (
            self.frozen_stages <= 4
        ), f"only 4 stages in ConvNeXt model, but got frozen_stages={self.frozen_stages}."

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, channel_last=False),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, channel_last=False),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, channel_last=False)
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()
        self._out_features = ["p{}".format(i) for i in self.out_indices]
        self._out_feature_channels = {"p{}".format(i): dims[i] for i in self.out_indices}
        self._out_feature_strides = {"p{}".format(i): 2 ** (i + 2) for i in self.out_indices}
        self._size_devisibility = 32

        self.apply(self._init_weights)

    def _freeze_stages(self):
        if self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages):
                # freeze downsample_layer's parameters
                downsampler_layer = self.downsample_layers[i]
                downsampler_layer.eval()
                for param in downsampler_layer.parameters():
                    param.requires_grad = False

                # freeze stage layer's parameters
                stage = self.stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs["p{}".format(i)] = x_out

        return outs

    def forward(self, x):
        """Forward function of `ConvNeXt`.

        Args:
            x (torch.Tensor): the input tensor for feature extraction.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p1") to tensor
        """
        x = self.forward_features(x)
        return x
