# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import List

from .structures import Instances


class MemoryBank(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        self.save_thresh = args.memory_bank_score_thresh
        self.save_period = 3
        self.max_his_length = args.memory_bank_len

        self.save_proj = nn.Linear(dim_in, dim_in)

        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)

        self.track_cls = nn.Linear(dim_in, 1)

        self.self_attn = None
        if args.memory_bank_with_self_attn:
            self.spatial_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
            self.spatial_fc1 = nn.Linear(dim_in, hidden_dim)
            self.spatial_fc2 = nn.Linear(hidden_dim, dim_in)
            self.spatial_norm1 = nn.LayerNorm(dim_in)
            self.spatial_norm2 = nn.LayerNorm(dim_in)
        else:
            self.spatial_attn = None

    def update(self, track_instances):
        embed = track_instances.output_embedding[:, None]  #( N, 1, 256)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            # saved_idxes = (save_period == 0)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:], torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_spatial_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        embed = track_instances.output_embedding
        dim = embed.shape[-1]
        query_pos = track_instances.query_pos[:, :dim]  # 应该为query_pos = pos2posemb(track_instances.ref_pts)
        k = q = (embed + query_pos)
        v = embed
        embed2 = self.spatial_attn(
            q[:, None],
            k[:, None],
            v[:, None]
        )[0][:, 0]
        embed = self.spatial_norm1(embed + embed2)
        embed2 = self.spatial_fc2(F.relu(self.spatial_fc1(embed)))
        embed = self.spatial_norm2(embed + embed2)
        track_instances.output_embedding = embed
        return track_instances

    def _forward_track_cls(self, track_instances):
        track_instances.track_scores = self.track_cls(track_instances.output_embedding)[..., 0]
        return track_instances

    def _forward_temporal_attn(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        dim = track_instances.query_pos.shape[1]
        key_padding_mask = track_instances.mem_padding_mask

        valid_idxes = key_padding_mask[:, -1] == 0
        embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            key_padding_mask = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],                  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),   # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
                key_padding_mask=key_padding_mask,
            )[0][0]

            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward_temporal_attn(self, track_instances):
        return self._forward_temporal_attn(track_instances)

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        if self.spatial_attn is not None:
            track_instances = self._forward_spatial_attn(track_instances)
        if self.track_cls is not None:
            track_instances = self._forward_track_cls(track_instances)
        return track_instances


def build_memory_bank(args, dim_in, hidden_dim, dim_out):
    name = args.memory_bank_type
    memory_banks = {
        'MemoryBank': MemoryBank,
    }
    assert name in memory_banks
    return memory_banks[name](args, dim_in, hidden_dim, dim_out)
