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

import pytest
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from detrex.modeling.backbone import TorchvisionBackbone


def test_torchvision_backbone():
    model_name = "resnet18"
    return_interm_indices = [0, 1, 2, 3]
    return_layers = {}
    for idx, layer_index in enumerate(return_interm_indices):
        return_layers.update(
            {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
        )

    # create backbone
    detrex_extractor = TorchvisionBackbone(model_name=model_name, return_nodes=return_layers)
    backbone = getattr(torchvision.models, model_name)()
    backbone.load_state_dict(detrex_extractor.model.state_dict())

    # torchvision extractor using IntermediateLayerGetter
    feature_extractor = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # input
    x = torch.randn(1, 3, 224, 224)
    outs_intermediatelayergetter = feature_extractor(x)
    outs_detrex = detrex_extractor(x)

    for layer_name, out_feature_name in return_layers.items():
        torch.allclose(
            outs_intermediatelayergetter[out_feature_name].sum(),
            outs_detrex[out_feature_name].sum(),
        )
