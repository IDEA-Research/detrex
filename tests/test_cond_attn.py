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

from ideadet.layers import ConditionalCrossAttention, ConditionalSelfAttention

from utils import OriginalConditionalAttentionDecoder, OriginalConditionalAttentionEncoder


def test_cond_self_attention():
    # hyper-parameters
    d_model = 256
    nhead = 8

    # module definition
    cond_attention_original = OriginalConditionalAttentionEncoder(d_model=d_model, nhead=nhead)
    cond_attention_ideadet = ConditionalSelfAttention(embed_dim=d_model, num_heads=nhead)

    # weight transfer
    cond_attention_ideadet.query_content_proj.weight = (
        cond_attention_original.sa_qcontent_proj.weight
    )
    cond_attention_ideadet.query_content_proj.bias = cond_attention_original.sa_qcontent_proj.bias
    cond_attention_ideadet.query_pos_proj.weight = cond_attention_original.sa_qpos_proj.weight
    cond_attention_ideadet.query_pos_proj.bias = cond_attention_original.sa_qpos_proj.bias
    cond_attention_ideadet.key_content_proj.weight = cond_attention_original.sa_kcontent_proj.weight
    cond_attention_ideadet.key_content_proj.bias = cond_attention_original.sa_kcontent_proj.bias
    cond_attention_ideadet.key_pos_proj.weight = cond_attention_original.sa_kpos_proj.weight
    cond_attention_ideadet.key_pos_proj.bias = cond_attention_original.sa_kpos_proj.bias
    cond_attention_ideadet.value_proj.weight = cond_attention_original.sa_v_proj.weight
    cond_attention_ideadet.value_proj.bias = cond_attention_original.sa_v_proj.bias
    cond_attention_ideadet.out_proj.weight = cond_attention_original.self_attn.out_proj.weight
    cond_attention_ideadet.out_proj.bias = cond_attention_original.self_attn.out_proj.bias

    # test output
    input = torch.randn(16, 1, 256)  # (n, b, c)
    query_pos = torch.randn(16, 1, 256)

    # self-attention with short-cut
    original_output = cond_attention_original(tgt=input, query_pos=query_pos)[0] + input

    ideadet_output = cond_attention_ideadet(
        query=input, key=input, value=input, query_pos=query_pos, key_pos=query_pos
    )

    torch.allclose(original_output.sum(), ideadet_output.sum())


def test_cond_decoder():
    # hyper-parameters
    d_model = 256
    nhead = 8

    # original conditional decoder
    cond_decoder_original = OriginalConditionalAttentionDecoder(
        d_model=d_model,
        nhead=nhead,
    )

    # ideadet self-attn + cross-attn
    ideadet_cond_self_attn = ConditionalSelfAttention(
        embed_dim=d_model,
        num_heads=nhead,
    )
    ideadet_cond_cross_attn = ConditionalCrossAttention(embed_dim=d_model, num_heads=nhead)

    # weight transfer
    ideadet_cond_self_attn.query_content_proj.weight = cond_decoder_original.sa_qcontent_proj.weight
    ideadet_cond_self_attn.query_content_proj.bias = cond_decoder_original.sa_qcontent_proj.bias
    ideadet_cond_self_attn.query_pos_proj.weight = cond_decoder_original.sa_qpos_proj.weight
    ideadet_cond_self_attn.query_pos_proj.bias = cond_decoder_original.sa_qpos_proj.bias
    ideadet_cond_self_attn.key_content_proj.weight = cond_decoder_original.sa_kcontent_proj.weight
    ideadet_cond_self_attn.key_content_proj.bias = cond_decoder_original.sa_kcontent_proj.bias
    ideadet_cond_self_attn.key_pos_proj.weight = cond_decoder_original.sa_kpos_proj.weight
    ideadet_cond_self_attn.key_pos_proj.bias = cond_decoder_original.sa_kpos_proj.bias
    ideadet_cond_self_attn.value_proj.weight = cond_decoder_original.sa_v_proj.weight
    ideadet_cond_self_attn.value_proj.bias = cond_decoder_original.sa_v_proj.bias
    ideadet_cond_self_attn.out_proj.weight = cond_decoder_original.self_attn.out_proj.weight
    ideadet_cond_self_attn.out_proj.bias = cond_decoder_original.self_attn.out_proj.bias

    ideadet_cond_cross_attn.query_content_proj.weight = (
        cond_decoder_original.ca_qcontent_proj.weight
    )
    ideadet_cond_cross_attn.query_content_proj.bias = cond_decoder_original.ca_qcontent_proj.bias
    ideadet_cond_cross_attn.query_pos_proj.weight = cond_decoder_original.ca_qpos_proj.weight
    ideadet_cond_cross_attn.query_pos_proj.bias = cond_decoder_original.ca_qpos_proj.bias
    ideadet_cond_cross_attn.key_content_proj.weight = cond_decoder_original.ca_kcontent_proj.weight
    ideadet_cond_cross_attn.key_content_proj.bias = cond_decoder_original.ca_kcontent_proj.bias
    ideadet_cond_cross_attn.key_pos_proj.weight = cond_decoder_original.ca_kpos_proj.weight
    ideadet_cond_cross_attn.key_pos_proj.bias = cond_decoder_original.ca_kpos_proj.bias
    ideadet_cond_cross_attn.value_proj.weight = cond_decoder_original.ca_v_proj.weight
    ideadet_cond_cross_attn.value_proj.bias = cond_decoder_original.ca_v_proj.bias
    ideadet_cond_cross_attn.out_proj.weight = cond_decoder_original.cross_attn.out_proj.weight
    ideadet_cond_cross_attn.out_proj.bias = cond_decoder_original.cross_attn.out_proj.bias
    ideadet_cond_cross_attn.query_pos_sine_proj.weight = (
        cond_decoder_original.ca_qpos_sine_proj.weight
    )
    ideadet_cond_cross_attn.query_pos_sine_proj.bias = cond_decoder_original.ca_qpos_sine_proj.bias

    # test output
    input = torch.randn(16, 1, 256)  # (n, b, c)
    query_pos = torch.randn(16, 1, 256)
    key_pos = torch.randn(16, 1, 256)
    query_sine_pos = torch.randn(16, 1, 256)

    original_output = cond_decoder_original(
        tgt=input, memory=input, query_pos=query_pos, pos=key_pos, query_sine_embed=query_sine_pos
    )

    # ideadet cond attn output
    temp = ideadet_cond_self_attn(
        query=input,
        key=input,
        value=input,
        query_pos=query_pos,
        key_pos=query_pos,
    )
    ideadet_output = ideadet_cond_cross_attn(
        query=temp,
        key=input,
        value=input,
        query_pos=query_pos,
        key_pos=key_pos,
        query_sine_embed=query_sine_pos,
        is_first_layer=True,
    )

    torch.allclose(original_output.sum(), ideadet_output.sum())
