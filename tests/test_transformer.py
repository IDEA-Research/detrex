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
# https://github.com/open-mmlab/mmcv/blob/master/tests/test_cnn/test_transformer.py
# ------------------------------------------------------------------------------------------------

import copy

import pytest
import torch
import torch.nn as nn

from ideadet.layers import (
    MultiheadAttention, 
    BaseTransformerLayer, 
    TransformerLayerSequence,
    FFN
)

def test_ffn():
    with pytest.raises(AssertionError):
        FFN(num_fcs=1)
    
    ffn = FFN(ffn_drop=0.)
    input_tensor = torch.rand(2, 20, 256)
    input_tensor_nbc = input_tensor.transpose(0, 1)
    assert torch.allclose(ffn(input_tensor).sum(), ffn(input_tensor_nbc).sum())
    residual = torch.rand_like(input_tensor)
    torch.allclose(
        ffn(input_tensor, identity=residual).sum(),
        ffn(input_tensor).sum() + residual.sum() - input_tensor.sum())


@pytest.mark.parametrize('embed_dim', [256])
def test_basetransformerlayer(embed_dim):
    attn = MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
    ffn = FFN(embed_dim, 1024, num_fcs=2, activation=nn.ReLU(inplace=True))
    base_layer = BaseTransformerLayer(
        attn=attn,
        ffn=ffn,
        norm=nn.LayerNorm(embed_dim),
        operation_order=('self_attn', 'norm', 'ffn', 'norm')
    )
    feedforward_dim = 1024

    assert attn.batch_first is True
    assert base_layer.ffns[0].feedforward_dim == feedforward_dim
    in_tensor = torch.rand(2, 10, embed_dim)
    base_layer(in_tensor)


def test_transformerlayersequence():
    sequence = TransformerLayerSequence(
        transformer_layers=BaseTransformerLayer(
            attn=[
                MultiheadAttention(256, 8, batch_first=True),
                MultiheadAttention(256, 8, batch_first=True)
            ],
            ffn=FFN(256, 1024, num_fcs=2),
            norm=nn.LayerNorm(256),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        ),
        num_layers=6
    )
    assert sequence.num_layers == 6
    assert sequence.pre_norm is False
    with pytest.raises(AssertionError):
        TransformerLayerSequence(
            transformer_layers=[
                BaseTransformerLayer(
                    attn=[
                        MultiheadAttention(256, 8, batch_first=True),
                        MultiheadAttention(256, 8, batch_first=True)
                    ],
                    ffn=FFN(256, 1024, num_fcs=2),
                    norm=nn.LayerNorm(256),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                ),
            ],
            num_layers=6
        )
