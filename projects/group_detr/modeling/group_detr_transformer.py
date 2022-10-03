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

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)

from .attention import GroupConditionalSelfAttention


class GroupDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
    ):
        super(GroupDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

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
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class GroupDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        group_nums: int = 11,
        num_layers: int = None,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
    ):
        super(GroupDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    GroupConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        group_nums=group_nums,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, 2, 2)

        self.bbox_embed = None

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

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
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(
            query_pos
        )  # [num_queries, batch_size, 2]
        reference_points: torch.Tensor = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 2]

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = get_sine_pos_embed(obj_center)
            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            query: torch.Tensor = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(1, 2),
                reference_points,
            ]

        return query.unsqueeze(0)


class GroupDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(GroupDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)

        hidden_state, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
        )

        return hidden_state, references
