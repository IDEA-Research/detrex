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

from __future__ import absolute_import, division, print_function
import unittest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from ideadet.layers.ms_deform_attn import MSDeformAttnFunction


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])


class TestMsDeformAttn(unittest.TestCase):
    def ms_deform_attn_core_pytorch(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        # for debug and test only,
        # need to use cuda version instead
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (
            (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .view(N_, M_ * D_, Lq_)
        )
        return output.transpose(1, 2).contiguous()

    @torch.no_grad()
    def test_forward_equal_with_pytorch_double(self):
        value = torch.rand(N, S, M, D).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 2
        output_pytorch = (
            self.ms_deform_attn_core_pytorch(
                value.double(), shapes, sampling_locations.double(), attention_weights.double()
            )
            .detach()
            .cpu()
        )
        output_cuda = (
            MSDeformAttnFunction.apply(
                value.double(),
                shapes,
                level_start_index,
                sampling_locations.double(),
                attention_weights.double(),
                im2col_step,
            )
            .detach()
            .cpu()
        )
        self.assertTrue(torch.allclose(output_cuda, output_pytorch))

