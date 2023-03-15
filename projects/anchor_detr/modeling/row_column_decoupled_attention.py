# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from nn.MultiheadAttention
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules import Module


def multi_head_rcda_forward(query_row: torch.Tensor,
                            query_col: torch.Tensor,
                            key_row: torch.Tensor,
                            key_col: torch.Tensor,
                            value: torch.Tensor,
                            embed_dim_to_check: int,
                            num_heads: int,
                            in_proj_weight: torch.Tensor,
                            in_proj_bias: torch.Tensor,
                            bias_k_row: Optional[torch.Tensor],
                            bias_k_col: Optional[torch.Tensor],
                            bias_v: Optional[torch.Tensor], 
                            add_zero_attn: bool,  # type: bool
                            dropout_p: float,  # type: float
                            out_proj_weight: torch.Tensor,  # type: Tensor
                            out_proj_bias: torch.Tensor,  # type: Tensor
                            training: bool=True,
                            key_padding_mask: Optional[torch.Tensor] =None,
                            need_weights: bool=True,  # type: bool
                            attn_mask: Optional[torch.Tensor]=None,
                            use_separate_proj_weight: bool=False,
                            q_row_proj_weight: Optional[torch.Tensor]=None,
                            q_col_proj_weight: Optional[torch.Tensor]=None,
                            k_row_proj_weight: Optional[torch.Tensor]=None,
                            k_col_proj_weight: Optional[torch.Tensor]=None,
                            v_proj_weight: Optional[torch.Tensor]=None,
                            static_k: Optional[torch.Tensor]=None,
                            static_v: Optional[torch.Tensor]=None
                            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    r"""
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight.
        q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
    """

    bsz, tgt_len, embed_dim = query_row.size()
    src_len_row = key_row.size()[2]
    src_len_col = key_col.size()[1]


    assert embed_dim == embed_dim_to_check
    # assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5


    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_row = F.linear(query_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 1
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_col = F.linear(query_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = embed_dim * 3
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_row = F.linear(key_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 3
    _end = embed_dim * 4
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_col = F.linear(key_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 4
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = F.linear(value, _w, _b)

    q_row = q_row.transpose(0, 1)
    q_col = q_col.transpose(0, 1)
    k_row = k_row.mean(1).transpose(0, 1)
    k_col = k_col.mean(2).transpose(0, 1)

    q_row = q_row * scaling
    q_col = q_col * scaling


    q_row = q_row.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q_col = q_col.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    if k_row is not None:
        k_row = k_row.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if k_col is not None:
        k_col = k_col.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().permute(1,2,0,3).reshape(src_len_col,src_len_row, bsz*num_heads, head_dim).permute(2,0,1,3)


    attn_output_weights_row = torch.bmm(q_row, k_row.transpose(1, 2))
    attn_output_weights_col = torch.bmm(q_col, k_col.transpose(1, 2))
    assert list(attn_output_weights_row.size()) == [bsz * num_heads, tgt_len, src_len_row]
    assert list(attn_output_weights_col.size()) == [bsz * num_heads, tgt_len, src_len_col]


    if key_padding_mask is not None:
        mask_row=key_padding_mask[:,0,:].unsqueeze(1).unsqueeze(2)
        mask_col=key_padding_mask[:,:,0].unsqueeze(1).unsqueeze(2)

        attn_output_weights_row = attn_output_weights_row.view(bsz, num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz, num_heads, tgt_len, src_len_col)

        attn_output_weights_row = attn_output_weights_row.masked_fill(mask_row,float('-inf'))
        attn_output_weights_col = attn_output_weights_col.masked_fill(mask_col, float('-inf'))

        attn_output_weights_row = attn_output_weights_row.view(bsz * num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz * num_heads, tgt_len, src_len_col)

    attn_output_weights_col = F.softmax(attn_output_weights_col, dim=-1)
    attn_output_weights_row = F.softmax(attn_output_weights_row, dim=-1)

    attn_output_weights_col = F.dropout(attn_output_weights_col, p=dropout_p, training=training)
    attn_output_weights_row = F.dropout(attn_output_weights_row, p=dropout_p, training=training)

    efficient_compute=True
    # This config will not affect the performance.
    # It will compute the short edge first which can save the memory and run slightly faster but both of them should get the same results.
    # You can also set it "False" if your graph needs to be always the same.
    if efficient_compute:
        if src_len_col<src_len_row:
            b_ein,q_ein,w_ein = attn_output_weights_row.shape
            b_ein,h_ein,w_ein,c_ein = v.shape
            attn_output_row = torch.matmul(attn_output_weights_row,v.permute(0,2,1,3).reshape(b_ein,w_ein,h_ein*c_ein)).reshape(b_ein,q_ein,h_ein,c_ein).permute(0,2,1,3)
            attn_output = torch.matmul(attn_output_weights_col.permute(1,0,2)[:,:,None,:],attn_output_row.permute(2,0,1,3)).squeeze(-2).reshape(tgt_len,bsz,embed_dim)
            ### the following code base on einsum get the same results
            # attn_output_row = torch.einsum("bqw,bhwc->bhqc",attn_output_weights_row,v)
            # attn_output = torch.einsum("bqh,bhqc->qbc",attn_output_weights_col,attn_output_row).reshape(tgt_len,bsz,embed_dim)
        else:
            b_ein,q_ein,h_ein=attn_output_weights_col.shape
            b_ein,h_ein,w_ein,c_ein = v.shape
            attn_output_col = torch.matmul(attn_output_weights_col,v.reshape(b_ein,h_ein,w_ein*c_ein)).reshape(b_ein,q_ein,w_ein,c_ein)
            attn_output = torch.matmul(attn_output_weights_row[:,:,None,:],attn_output_col).squeeze(-2).permute(1,0,2).reshape(tgt_len, bsz, embed_dim)
            ### the following code base on einsum get the same results
            # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
            # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)
    else:
        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
        attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        ### the following code base on einsum get the same results
        # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
        # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        return attn_output,torch.einsum("bqw,bqh->qbhw",attn_output_weights_row,attn_output_weights_col).reshape(tgt_len,bsz,num_heads,src_len_col,src_len_row).mean(2)
    else:
        return attn_output, None



class MultiheadRCDA(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference:
        Anchor DETR: Query Design for Transformer-Based Detector

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::
        >>> multihead_attn = MultiheadRCDA(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query_row, query_col, key_row, key_col, value)
    """
    __annotations__ = {
        'bias_k_row': torch._jit_internal.Optional[torch.Tensor],
        'bias_k_col': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_row_proj_weight', 'q_col_proj_weight', 'k_row_proj_weight', 'k_col_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            dropout=0., 
            bias=True, 
            add_bias_kv=False, 
            add_zero_attn=False, 
            kdim=None, 
            vdim=None
        ):
        super(MultiheadRCDA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_row_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.q_col_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_row_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.k_col_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(5 * embed_dim, embed_dim))
            self.register_parameter('q_row_proj_weight', None)
            self.register_parameter('q_col_proj_weight', None)
            self.register_parameter('k_row_proj_weight', None)
            self.register_parameter('k_col_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(5 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k_row = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_k_col = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k_row = self.bias_k_col = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            init.xavier_uniform_(self.in_proj_weight)
        else:
            init.xavier_uniform_(self.q_row_proj_weight)
            init.xavier_uniform_(self.q_col_proj_weight)
            init.xavier_uniform_(self.k_row_proj_weight)
            init.xavier_uniform_(self.k_col_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.)
            init.constant_(self.out_proj.bias, 0.)
        if self.bias_k_row is not None:
            init.xavier_normal_(self.bias_k_row)
        if self.bias_k_col is not None:
            init.xavier_normal_(self.bias_k_col)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadRCDA, self).__setstate__(state)

    def forward(
            self, 
            query_row, 
            query_col, 
            key_row, 
            key_col, 
            value,
            key_padding_mask=None, 
            need_weights=False, 
            attn_mask=None
        ):

        r"""
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_rcda_forward(
                query_row,query_col, key_row, key_col, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k_row,self.bias_k_col, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_row_proj_weight=self.q_row_proj_weight, q_col_proj_weight=self.q_col_proj_weight,
                k_row_proj_weight=self.k_row_proj_weight, k_col_proj_weight=self.k_col_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_rcda_forward(
                query_row,query_col, key_row,key_col, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k_row,self.bias_k_col, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

