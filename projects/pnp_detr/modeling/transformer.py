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

import random

import torch
import torch.nn as nn

from detrex.layers import FFN, BaseTransformerLayer, MultiheadAttention, TransformerLayerSequence


class PnPDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = True,
        batch_first: bool = False,
    ):
        super(PnPDetrTransformerEncoder, self).__init__(
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
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
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
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class PnPDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = True,
        return_intermediate: bool = True,
        batch_first: bool = False,
    ):
        super(PnPDetrTransformerDecoder, self).__init__(
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

        if not self.return_intermediate:
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

            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)[None]
            return query

        # return intermediate
        intermediate = []
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

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        return torch.stack(intermediate)


class SortSampler(nn.Module):
    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc', kproj_net='2layer-fc', unsample_abstract_number=30,pos_embed_kproj=False):
        super().__init__()
        self.topk_ratio = topk_ratio
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-16':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 16, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(16, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.unsample_abstract_number = unsample_abstract_number
        if kproj_net == '2layer-fc':
            self.k_proj = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, unsample_abstract_number))
        elif kproj_net == '1layer-fc':
            self.k_proj = nn.Linear(input_dim, unsample_abstract_number)
        else:
            raise ValueError
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed_kproj = pos_embed_kproj

    def forward(self, src, mask, pos_embed, sample_ratio):
        bs,c ,h, w  = src.shape
        sample_weight = self.score_pred_net(src).sigmoid().view(bs,-1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()
        sample_weight_clone[mask] = -1.

        if sample_ratio==None:
            sample_ratio = self.topk_ratio
        ##max sample number:
        sample_lens = ((~mask).sum(1)*sample_ratio).int()
        max_sample_num = sample_lens.max()
        mask_topk = torch.arange(max_sample_num).expand(len(sample_lens), max_sample_num).to(sample_lens.device) > (sample_lens-1).unsqueeze(1)

        ## for sampling remaining unsampled points
        min_sample_num = sample_lens.min()

        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]
        sort_confidence_topk = sort_order[:,:max_sample_num]
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        src_sample_remaining = src.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))

        ## this will maskout the padding and sampled points
        mask_unsampled = torch.arange(mask.size(1)).expand(len(sample_lens), mask.size(1)).to(sample_lens.device) < (sample_lens).unsqueeze(1)
        mask_unsampled = mask_unsampled | mask.gather(1, sort_order)
        mask_unsampled = mask_unsampled[:,min_sample_num:]

        ## abstract the unsampled points with attention
        if self.pos_embed_kproj:
            pos_embed_sample_remaining = pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
            kproj = self.k_proj(src_sample_remaining+pos_embed_sample_remaining)
        else:
            kproj = self.k_proj(src_sample_remaining)
        kproj = kproj.masked_fill(
            mask_unsampled.permute(1,0).unsqueeze(2),
            float('-inf'),
        ).permute(1,2,0).softmax(-1)
        abs_unsampled_points = torch.bmm(kproj, self.v_proj(src_sample_remaining).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_pos_embed = torch.bmm(kproj, pos_embed.gather(0,sort_confidence_topk_remaining.
                                                                          permute(1,0)[...,None].expand(-1,-1,c)).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_mask = mask.new_zeros(mask.size(0),abs_unsampled_points.size(0))

        ## reg sample weight to be sparse with l1 loss
        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()
        src_sampled = src.gather(0, sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) *sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
        pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        mask_sampled = mask_topk

        src = torch.cat([src_sampled, abs_unsampled_points])
        pos_embed = torch.cat([pos_embed_sampled,abs_unsampled_pos_embed])
        mask = torch.cat([mask_sampled, abs_unsampled_mask],dim=1)
        assert ((~mask).sum(1)==sample_lens+self.unsample_abstract_number).all()
        return src, sample_reg_loss, sort_confidence_topk, mask, pos_embed


class PnPDetrTransformer(nn.Module):
    def __init__(
        self, 
        encoder=None, 
        decoder=None,
        sample_topk_ratio=1/3.,
        score_pred_net='2layer-fc-256',
        kproj_net='2layer-fc',
        unsample_abstract_number=30,
        pos_embed_kproj=False,
    ):
        super(PnPDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.sampler = SortSampler(
            sample_topk_ratio, 
            self.embed_dim, 
            score_pred_net=score_pred_net, 
            kproj_net=kproj_net, 
            unsample_abstract_number=unsample_abstract_number, 
            pos_embed_kproj=pos_embed_kproj
        )

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed, sample_ratio):
        bs, c, h, w = x.shape
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1
        )  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        x, sample_reg_loss, sort_confidence_topk, mask, pos_embed = self.sampler(x, mask, pos_embed, sample_ratio)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        decoder_output = decoder_output.transpose(1, 2)
        return decoder_output, sample_reg_loss
