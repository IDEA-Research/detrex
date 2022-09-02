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


import torchvision
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)
from torchvision.models._utils import IntermediateLayerGetter
from detectron2.modeling.backbone import Backbone

class TorchvisionBackbone(Backbone):
    def __init__(self,
                 model_name: str = "resnet50",
                 return_nodes: dict = {
                    "layer1": "res2",
                    "layer2": "res3",
                    "layer3": "res4",
                    "layer4": "res5",
                 },
                 **kwargs,
                ):
        super(TorchvisionBackbone, self).__init__()
        self.model = getattr(torchvision.models, model_name)(**kwargs)
        self.feature_extractor = create_feature_extractor(
            model = self.model,
            return_nodes=return_nodes
        )
    
    def forward(self, x):
        outs = self.feature_extractor(x)
        return outs