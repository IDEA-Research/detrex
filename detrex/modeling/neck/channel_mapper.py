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
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/channel_mapper.py
# ------------------------------------------------------------------------------------------------

from typing import Dict, List
import torch.nn as nn

from detrex.layers import ConvNormAct

from detectron2.modeling import ShapeSpec


class ChannelMapper(nn.Module):
    def __init__(
        self,
        input_shapes: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: nn.Module = None,
        activation: nn.Module = None,
        num_outs: int = None,
        **kwargs,
    ):
        super(ChannelMapper, self).__init__()
        self.extra_convs = None

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        if num_outs is None:
            num_outs = len(input_shapes)

        self.convs = nn.ModuleList()
        for in_channel in in_channels_per_feature:
            self.convs.append(
                ConvNormAct(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvNormAct(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=bias,
                        groups=groups,
                        dilation=dilation,
                        norm_layer=norm_layer,
                        activation=activation,
                    )
                )

        self.input_shapes = input_shapes
        self.in_features = in_features
        self.out_channels = out_channels

    def forward(self, inputs):
        # inputs: key, value
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
