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

import pytest
import torch

from ideadet.layers import FFN

from utils import MLP


def test_ffn_output():
    embed_dim = 256
    feedforward_dim = 256
    output_dim = 4
    num_fcs = 3

    mlp_layer = MLP(
        input_dim=embed_dim, hidden_dim=feedforward_dim, output_dim=output_dim, num_layers=num_fcs
    )

    ffn_layer = FFN(
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
        output_dim=output_dim,
        num_fcs=num_fcs,
        add_identity=False,
    )

    # transfer weight
    ffn_layer.layers[0][0].weight = mlp_layer.layers[0].weight
    ffn_layer.layers[0][0].bias = mlp_layer.layers[0].bias
    ffn_layer.layers[1][0].weight = mlp_layer.layers[1].weight
    ffn_layer.layers[1][0].bias = mlp_layer.layers[1].bias
    ffn_layer.layers[2].weight = mlp_layer.layers[2].weight
    ffn_layer.layers[2].bias = mlp_layer.layers[2].bias

    # test output
    x = torch.randn(16, 256)
    mlp_output = mlp_layer(x)
    ffn_output = ffn_layer(x)
    torch.allclose(mlp_output.sum(), ffn_output.sum())
