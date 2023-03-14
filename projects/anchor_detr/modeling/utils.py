# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import math

import torch


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed
