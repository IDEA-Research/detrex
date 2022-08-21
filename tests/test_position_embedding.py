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
# https://github.com/open-mmlab/mmdetection/blob/master/tests/test_models/test_utils/test_position_encoding.py
# ------------------------------------------------------------------------------------------------


import pytest
import torch

from ideadet.layers import (
    PositionEmbeddingSine,
    PositionEmbeddingLearned,
)

from utils import (
    DABPositionEmbeddingLearned,
    DABPositionEmbeddingSine,
    DeformablePositionEmbeddingSine,
)

def test_sine_position_embedding(num_pos_feats=16, batch_size=2):
    # test invalid type of scale
    with pytest.raises(AssertionError):
        module = PositionEmbeddingSine(
            num_pos_feats, scale=(3., ), normalize=True)
    
    module = PositionEmbeddingSine(num_pos_feats)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)
    assert not module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)

    # set normalize
    module = PositionEmbeddingSine(num_pos_feats, normalize=True)
    assert module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)


def test_learned_position_embedding(num_pos_feats=16,
                                     row_num_embed=10,
                                     col_num_embed=10,
                                     batch_size=2):
    module = PositionEmbeddingLearned(num_pos_feats, row_num_embed, col_num_embed)
    assert module.row_embed.weight.shape == (row_num_embed, num_pos_feats)
    assert module.col_embed.weight.shape == (col_num_embed, num_pos_feats)
    h, w = 10, 6
    mask = torch.rand(batch_size, h, w) > 0.5
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)


def test_sine_position_embedding_output(num_pos_feats=16, batch_size=2):
    # test position embedding without normalize
    module_new = PositionEmbeddingSine(num_pos_feats)
    module_original = DABPositionEmbeddingSine(num_pos_feats)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)

    output_new = module_new(mask)
    output_original = module_original(mask)

    torch.allclose(output_new.sum(), output_original.sum())

    # test position embedding with normalize
    module_new = PositionEmbeddingSine(num_pos_feats, normalize=True)
    module_original = DABPositionEmbeddingSine(num_pos_feats, normalize=True)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)

    output_new = module_new(mask)
    output_original = module_original(mask)

    torch.allclose(output_new.sum(), output_original.sum())


def test_sine_position_embedding_deformable(num_pos_feats=16, batch_size=2):
    # test position embedding used in Deformable-DETR without normalize
    # test position embedding without normalize
    module_new = PositionEmbeddingSine(num_pos_feats, offset=-0.5)
    module_original = DeformablePositionEmbeddingSine(num_pos_feats)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)

    output_new = module_new(mask)
    output_original = module_original(mask)

    torch.allclose(output_new.sum(), output_original.sum())

    # test position embedding with normalize
    module_new = PositionEmbeddingSine(num_pos_feats, offset=-0.5, normalize=True)
    module_original = DeformablePositionEmbeddingSine(num_pos_feats, normalize=True)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)

    output_new = module_new(mask)
    output_original = module_original(mask)

    torch.allclose(output_new.sum(), output_original.sum())