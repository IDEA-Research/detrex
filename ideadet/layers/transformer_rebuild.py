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
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py
# ------------------------------------------------------------------------------------------------


import copy
import warnings
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=True,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.proj_drop(out)


class FFN(nn.Module):
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

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class BaseTransformerLayer(nn.Module):
    def __init__(self, attn, ffn: nn.Module, norm: nn.Module, operation_order: dict = None):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order) == {"self_attn", "norm", "cross_attn", "ffn"}

        # count attention nums
        self.num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"

        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    **kwargs,
                )

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

            return query


class TransformerLayerSequence(nn.Module):
    def __init__(
        self,
        transformer_layers=None,
        num_layers=None,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(copy.deepcopy(transformer_layers))
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query
