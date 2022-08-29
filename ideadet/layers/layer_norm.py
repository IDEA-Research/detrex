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
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm which supports both channel_last (default) and channel_first data format.
    The inputs data format should be as follows:
        - channel_last: (bs, h, w, channels)
        - channel_first: (bs, channels, h, w)

    Args:
        normalized_shape (tuple): The size of the input feature dim.
        eps (float): A value added to the denominator for 
            numerical stability. Default: True.
        channel_last (bool): Set True for `channel_last` input data 
            format. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6, channel_last=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_last = channel_last
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Forward function for `LayerNorm`
        """
        if self.channel_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
