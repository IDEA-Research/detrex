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
# Modified from:
# https://github.com/FrancescoSaverioZuppichini/glasses/blob/master/glasses/nn/blocks/__init__.py
# ------------------------------------------------------------------------------------------------

from functools import partial
import torch.nn as nn


class ConvNormAct(nn.Module):
    """Utility module that stacks one convolution 2D layer, 
    a normalization layer and an activation function.
    
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Size of the convolving kernel. Default: 1.
        stride (int): Stride of convolution. Default: 1.
        padding (int): Padding added to all four sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels 
            to output channels. Default: 1.
        bias (bool): if True, adds a learnable bias to the output. Default: True.
        norm_layer (nn.Module): Normalization layer used in `ConvNormAct`. Default: None.
        activation (nn.Module): Activation layer used in `ConvNormAct`. Default: None.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_layer: nn.Module = None,
        activation: nn.Module = None,
        **kwargs,
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.norm = norm_layer
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


ConvNorm = partial(ConvNormAct, activation=None)