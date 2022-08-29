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

from typing import Tuple
import torch
import torch.nn as nn

from ideadet.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from ideadet.utils import inverse_sigmoid


class DabDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        operation_order: Tuple = ("self_attn", "norm", "ffn", "norm"),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
    ):
        super(DabDetrTransformerEncoder, self).__init__(
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
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

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
            position_scales = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        operation_order: Tuple = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        num_layers: int = None,
        query_dim: int = 4,
        modulate_hw_attn: bool = True,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
    ):
        super(DabDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
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
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(
            query_dim // 2 * self.embed_dim, self.embed_dim, self.embed_dim, 2
        )

        self.bbox_embed = None

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

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
        refpoints_embed=None,
        **kwargs,
    ):
        intermediate = []

        reference_points = refpoints_embed.sigmoid()
        refpoints = [reference_points]

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
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

            # iter update
            if self.bbox_embed is not None:
                temp = self.bbox_embed(query)
                temp[..., : self.embed_dim] += inverse_sigmoid(reference_points)
                new_reference_points = temp[..., : self.embed_dim].sigmoid()

                if idx != self.num_layers - 1:
                    refpoints.append(new_reference_points)
                reference_points = new_reference_points.detach()

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
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(refpoints).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                ]

        return query.unsqueeze(0)


class DabDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, as_two_stage=False, two_stage_num_proposals=300):
        super(DabDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """generate proposals from encoder output

        """
        memory = memory.transpose(0, 1) # bs, num_token, dim
                                        # memory_padding_mask: bs, dim

        N, S, C = memory.shape
        H, W = spatial_shapes
        proposals = []
        mask_flatten_ = memory_padding_mask.view(N, H, W, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05
        proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
        proposals.append(proposal)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, x, mask, refpoints_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )

        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask, (h, w)
            ) 

            enc_outputs_class = self.enc_class_embed(output_memory)
            enc_outputs_coord_unact = (
                self.enc_bbox_embed(output_memory) + output_proposals
            )
            # output_memory: bs, num_tokens, c
            # output_proposals: bs, num_tokens, 4

            topk = self.two_stage_num_proposals
            if topk > (h*w):
                topk_coords_unact = enc_outputs_coord_unact
                reference_points = topk_coords_unact.sigmoid()
                refpoints_embed = topk_coords_unact.transpose(0, 1).detach() # num_queries, bs, 4. unsigmoid.
                target = output_memory.transpose(0, 1).detach() # num_queries, bs, embed_dim

                # pad to num_queries
                pad_num = self.two_stage_num_proposals - refpoints_embed.shape[0]
                pad_anchors = inverse_sigmoid(torch.rand(pad_num, bs, 4, device=refpoints_embed.device))
                pad_tgt = torch.zeros(pad_num, bs, self.embed_dim, device=refpoints_embed.device)

                # init states for decoder
                refpoints_embed = torch.cat([refpoints_embed, pad_anchors], 0)
                target = torch.cat([target, pad_tgt], 0)

                # states for return
                target_unact = torch.cat((output_memory, pad_tgt.transpose(0, 1)), 1) # bs, num_tokens, embed_dim
                reference_points = torch.cat((reference_points, pad_anchors.sigmoid().transpose(0, 1)), 1) # bs, num_tokens, 4

            else:
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
                # extract region proposal boxes
                topk_coords_unact = torch.gather(
                    enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )
                reference_points = topk_coords_unact.sigmoid()
                refpoints_embed = topk_coords_unact.transpose(0, 1).detach() # num_queries, bs, 4. unsigmoid.

                # extract region features
                target_unact = torch.gather(
                    output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
                )
                target = target_unact.transpose(0, 1).detach() # num_queries, bs, embed_dim
        else:
            refpoints_embed = refpoints_embed.unsqueeze(1).repeat(1, bs, 1) # num_queries, bs, 4. unsigmoid.
            num_queries = refpoints_embed.shape[0]
            target = torch.zeros(num_queries, bs, self.embed_dim, device=refpoints_embed.device)

        hidden_state, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            refpoints_embed=refpoints_embed,
        )

        if self.as_two_stage:
            return hidden_state, references, target_unact, reference_points

        return hidden_state, references, None, None
