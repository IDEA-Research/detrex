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

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The implementation of simple multi-layer perceptron layer
    without dropout and identity connection.

    The feature process order follows `Linear -> ReLU -> Linear -> ReLU -> ...`.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden dimension of MLPs.
        output_dim (int): the output feature dimension of MLPs.
        num_layer (int): The number of FC layer used in MLPs.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> torch.Tensor:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):
    """The implementation of feed-forward networks (FFNs)
    with identity connection.

    Args:
        embed_dim (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_dim (int): The hidden dimension of FFNs.
            Defaults: 1024.
        output_dim (int): The output feature dimension of FFNs.
            Default: None. If None, the `embed_dim` will be used.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        activation (nn.Module): The activation layer used in FFNs.
            Default: nn.ReLU(inplace=True).
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """

    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        output_dim=None,
        num_fcs=2,
        activation=nn.ReLU(inplace=True),
        ffn_drop=0.0,
        fc_bias=True,
        add_identity=True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None) -> torch.Tensor:
        """Forward function of `FFN`.

        Args:
            x (torch.Tensor): the input tensor used in `FFN` layers.
            identity (torch.Tensor): the tensor with the same shape as `x`,
                which will be used for identity addition. Default: None.
                if None, `x` will be used.

        Returns:
            torch.Tensor: the forward results of `FFN` layer
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out
