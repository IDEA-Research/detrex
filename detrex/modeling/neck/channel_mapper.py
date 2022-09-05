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

import copy
from typing import Dict, List
import torch.nn as nn

from detrex.layers import ConvNormAct

from detectron2.modeling import ShapeSpec


class ChannelMapper(nn.Module):
    """Channel Mapper for reduce/increase channels of backbone features. Modified
    from `mmdet <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/channel_mapper.py>`_.

    This is used to reduce/increase the channels of backbone features.

    Args:
        input_shape (Dict[str, ShapeSpec]): A dict which contains the backbone features meta infomation,
            e.g. ``input_shape = {"res5": ShapeSpec(channels=2048)}``.
        in_features (List[str]): A list contains the keys which maps the features output from the backbone,
            e.g. ``in_features = ["res"]``.
        out_channels (int): Number of output channels for each scale.
        kernel_size (int, optional): Size of the convolving kernel for each scale.
            Default: 3.
        stride (int, optional): Stride of convolution for each scale. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output of each scale.
            Default: True.
        groups (int, optional): Number of blocked connections from input channels to
            output channels for each scale. Default: 1.
        dilation (int, optional): Spacing between kernel elements for each scale.
            Default: 1.
        norm_layer (nn.Module, optional): The norm layer used for each scale. Default: None.
        activation (nn.Module, optional): The activation layer used for each scale. Default: None.
        num_outs (int, optional): Number of output feature maps. There will be ``extra_convs`` when
            ``num_outs`` is larger than the length of ``in_features``. Default: None.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from detrex.modeling import ChannelMapper
        >>> from detectron2.modeling import ShapeSpec
        >>> input_features = {
        ... "p0": torch.randn(1, 128, 128, 128),
        ... "p1": torch.randn(1, 256, 64, 64),
        ... "p2": torch.randn(1, 512, 32, 32),
        ... "p3": torch.randn(1, 1024, 16, 16),
        ... }
        >>> input_shapes = {
        ... "p0": ShapeSpec(channels=128),
        ... "p1": ShapeSpec(channels=256),
        ... "p2": ShapeSpec(channels=512),
        ... "p3": ShapeSpec(channels=1024),
        ... }
        >>> in_features = ["p0", "p1", "p2", "p3"]
        >>> neck = ChannelMapper(
        ... input_shapes=input_shapes,
        ... in_features=in_features,
        ... out_channels=256,
        ... norm_layer=nn.GroupNorm(num_groups=32, num_channels=256)
        >>> outputs = neck(input_features)
        >>> for i in range(len(outputs)):
        ... print(f"output[{i}].shape = {outputs[i].shape}")
        output[0].shape = torch.Size([1, 256, 128, 128])
        output[1].shape = torch.Size([1, 256, 64, 64])
        output[2].shape = torch.Size([1, 256, 32, 32])
        output[3].shape = torch.Size([1, 256, 16, 16])
    """

    def __init__(
        self,
        input_shapes: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 3,
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
                    norm_layer=copy.deepcopy(norm_layer),
                    activation=copy.deepcopy(activation),
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
                        norm_layer=copy.deepcopy(norm_layer),
                        activation=copy.deepcopy(activation),
                    )
                )

        self.input_shapes = input_shapes
        self.in_features = in_features
        self.out_channels = out_channels

    def forward(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (Dict[str, torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
