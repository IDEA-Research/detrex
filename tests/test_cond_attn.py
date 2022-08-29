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

from detrex.layers import ConditionalCrossAttention, ConditionalSelfAttention

from utils import OriginalConditionalAttentionDecoder, OriginalConditionalAttentionEncoder


def test_cond_self_attention():
    # hyper-parameters
    d_model = 256
    nhead = 8

    # module definition
    cond_attention_original = OriginalConditionalAttentionEncoder(d_model=d_model, nhead=nhead)
    cond_attention_detrex = ConditionalSelfAttention(embed_dim=d_model, num_heads=nhead)

    # weight transfer
    cond_attention_detrex.query_content_proj.weight = (
        cond_attention_original.sa_qcontent_proj.weight
    )
    cond_attention_detrex.query_content_proj.bias = cond_attention_original.sa_qcontent_proj.bias
    cond_attention_detrex.query_pos_proj.weight = cond_attention_original.sa_qpos_proj.weight
    cond_attention_detrex.query_pos_proj.bias = cond_attention_original.sa_qpos_proj.bias
    cond_attention_detrex.key_content_proj.weight = cond_attention_original.sa_kcontent_proj.weight
    cond_attention_detrex.key_content_proj.bias = cond_attention_original.sa_kcontent_proj.bias
    cond_attention_detrex.key_pos_proj.weight = cond_attention_original.sa_kpos_proj.weight
    cond_attention_detrex.key_pos_proj.bias = cond_attention_original.sa_kpos_proj.bias
    cond_attention_detrex.value_proj.weight = cond_attention_original.sa_v_proj.weight
    cond_attention_detrex.value_proj.bias = cond_attention_original.sa_v_proj.bias
    cond_attention_detrex.out_proj.weight = cond_attention_original.self_attn.out_proj.weight
    cond_attention_detrex.out_proj.bias = cond_attention_original.self_attn.out_proj.bias

    # test output
    input = torch.randn(16, 1, 256)  # (n, b, c)
    query_pos = torch.randn(16, 1, 256)

    # self-attention with short-cut
    original_output = cond_attention_original(tgt=input, query_pos=query_pos)[0] + input

    detrex_output = cond_attention_detrex(
        query=input, key=input, value=input, query_pos=query_pos, key_pos=query_pos
    )

    torch.allclose(original_output.sum(), detrex_output.sum())


def test_cond_decoder():
    # hyper-parameters
    d_model = 256
    nhead = 8

    # original conditional decoder
    cond_decoder_original = OriginalConditionalAttentionDecoder(
        d_model=d_model,
        nhead=nhead,
    )

    # detrex self-attn + cross-attn
    detrex_cond_self_attn = ConditionalSelfAttention(
        embed_dim=d_model,
        num_heads=nhead,
    )
    detrex_cond_cross_attn = ConditionalCrossAttention(embed_dim=d_model, num_heads=nhead)

    # weight transfer
    detrex_cond_self_attn.query_content_proj.weight = cond_decoder_original.sa_qcontent_proj.weight
    detrex_cond_self_attn.query_content_proj.bias = cond_decoder_original.sa_qcontent_proj.bias
    detrex_cond_self_attn.query_pos_proj.weight = cond_decoder_original.sa_qpos_proj.weight
    detrex_cond_self_attn.query_pos_proj.bias = cond_decoder_original.sa_qpos_proj.bias
    detrex_cond_self_attn.key_content_proj.weight = cond_decoder_original.sa_kcontent_proj.weight
    detrex_cond_self_attn.key_content_proj.bias = cond_decoder_original.sa_kcontent_proj.bias
    detrex_cond_self_attn.key_pos_proj.weight = cond_decoder_original.sa_kpos_proj.weight
    detrex_cond_self_attn.key_pos_proj.bias = cond_decoder_original.sa_kpos_proj.bias
    detrex_cond_self_attn.value_proj.weight = cond_decoder_original.sa_v_proj.weight
    detrex_cond_self_attn.value_proj.bias = cond_decoder_original.sa_v_proj.bias
    detrex_cond_self_attn.out_proj.weight = cond_decoder_original.self_attn.out_proj.weight
    detrex_cond_self_attn.out_proj.bias = cond_decoder_original.self_attn.out_proj.bias

    detrex_cond_cross_attn.query_content_proj.weight = (
        cond_decoder_original.ca_qcontent_proj.weight
    )
    detrex_cond_cross_attn.query_content_proj.bias = cond_decoder_original.ca_qcontent_proj.bias
    detrex_cond_cross_attn.query_pos_proj.weight = cond_decoder_original.ca_qpos_proj.weight
    detrex_cond_cross_attn.query_pos_proj.bias = cond_decoder_original.ca_qpos_proj.bias
    detrex_cond_cross_attn.key_content_proj.weight = cond_decoder_original.ca_kcontent_proj.weight
    detrex_cond_cross_attn.key_content_proj.bias = cond_decoder_original.ca_kcontent_proj.bias
    detrex_cond_cross_attn.key_pos_proj.weight = cond_decoder_original.ca_kpos_proj.weight
    detrex_cond_cross_attn.key_pos_proj.bias = cond_decoder_original.ca_kpos_proj.bias
    detrex_cond_cross_attn.value_proj.weight = cond_decoder_original.ca_v_proj.weight
    detrex_cond_cross_attn.value_proj.bias = cond_decoder_original.ca_v_proj.bias
    detrex_cond_cross_attn.out_proj.weight = cond_decoder_original.cross_attn.out_proj.weight
    detrex_cond_cross_attn.out_proj.bias = cond_decoder_original.cross_attn.out_proj.bias
    detrex_cond_cross_attn.query_pos_sine_proj.weight = (
        cond_decoder_original.ca_qpos_sine_proj.weight
    )
    detrex_cond_cross_attn.query_pos_sine_proj.bias = cond_decoder_original.ca_qpos_sine_proj.bias

    # test output
    input = torch.randn(16, 1, 256)  # (n, b, c)
    query_pos = torch.randn(16, 1, 256)
    key_pos = torch.randn(16, 1, 256)
    query_sine_pos = torch.randn(16, 1, 256)

    original_output = cond_decoder_original(
        tgt=input, memory=input, query_pos=query_pos, pos=key_pos, query_sine_embed=query_sine_pos
    )

    # detrex cond attn output
    temp = detrex_cond_self_attn(
        query=input,
        key=input,
        value=input,
        query_pos=query_pos,
        key_pos=query_pos,
    )
    detrex_output = detrex_cond_cross_attn(
        query=temp,
        key=input,
        value=input,
        query_pos=query_pos,
        key_pos=key_pos,
        query_sine_embed=query_sine_pos,
        is_first_layer=True,
    )

    torch.allclose(original_output.sum(), detrex_output.sum())
