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


class BaseTransformerLayer(nn.Module):
    def __init__(self, attn, ffn: nn.Module, norm: nn.Module, operation_order: tuple = None):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, f"The length of attn (nn.Module or List[nn.Module]) {num_attn}" \
                f"is not consistent with the number of attention in operation_order {operation_order}"

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

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
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

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


class DetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        post_norm=True,
        **kwargs,
    ):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
    
    def forward(self, query, *args, **kwargs):
        x = super(DetrTransformerEncoder, self).forward(query, *args, **kwargs)
        if self.post_norm_layer is not None:
            x = self.post_norm_layer(x)
        return x


class DetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        post_norm=True,
        return_intermediate=True,
        **kwargs
    ):
        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
    

    def forward(self, query, *args, **kwargs):
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm_layer is not None:
                x = self.post_norm(x)[None]
            return x
        
        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


class DetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(DetrTransformer, self).__init__()
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
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
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
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return decoder_output, memory