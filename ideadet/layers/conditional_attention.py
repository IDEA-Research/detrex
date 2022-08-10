import  warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.,
        proj_drop=0.,
        batch_first=False,
        **kwargs,
    ):
        super(ConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
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
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        
        assert query_pos is not None and key_pos is not None, "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        # attention calculation
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop=0.,
        proj_drop=0.,
        batch_first=False,
        is_first_layer=False,
        **kwargs,
    ):
        super(ConditionalCrossAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first
        self.is_first_layer = is_first_layer

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, query_sine_embed=None, attn_mask=None, key_padding_mask=None, **kwargs):
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
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        
        assert query_pos is not None and key_pos is not None, "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        N, B, C = query_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if self.is_first_layer:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        # preprocess
        q = q.view(N, B, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(N, B, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)
        k = k.view(N, B, self.num_heads, C // self.num_heads)
        key_pos = key_pos.view(N, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(N, B, C * 2)
        

        # attention calculation
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)