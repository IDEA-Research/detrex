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

from typing import Any, Dict
import torchvision

from detrex.modeling.backbone import Backbone

try:
    from torchvision.models.feature_extraction import (
        create_feature_extractor,
    )
    has_feature_extractor = True
except ImportError:
    has_feature_extractor = False


class TorchvisionBackbone(Backbone):
    """A wrapper for torchvision pretrained backbones
    
    Please check `Feature extraction for model inspection
    <https://pytorch.org/vision/stable/feature_extraction.html>`_
    for more details.

    Args:
        model_name (str): Name of torchvision models. Default: resnet50.
        pretrained (bool): Whether to load pretrained weights. Default: False.
        weights (Optional[ResNet50_Weights]): The pretrained weights to use. Default: None.
        return_nodes (Dict[str, str]): The keys are the node names and the values are the
            user-specified keys for the graph module's returned dictionary.
    """
    def __init__(self,
                 model_name: str = "resnet50",
                 pretrained: bool = False,
                 return_nodes: Dict[str, str] = {
                    "layer1": "res2",
                    "layer2": "res3",
                    "layer3": "res4",
                    "layer4": "res5",
                 },
                 train_return_nodes: Dict[str, str] = None,
                 eval_return_nodes: Dict[str, str] = None,
                 tracer_kwargs: Dict[str, Any] = None,
                 suppress_diff_warnings: bool = False,
                 **kwargs,
                ):
        super(TorchvisionBackbone, self).__init__()
        
        # build torchvision models
        self.model = getattr(torchvision.models, model_name)(
            pretrained=pretrained,
            **kwargs
        )
        
        if has_feature_extractor is False:
            raise RuntimeError('Failed to import create_feature_extractor from torchvision. \
            Please install torchvision 1.10+.')
        
        # turn models into feature extractor
        self.feature_extractor = create_feature_extractor(
            model = self.model,
            return_nodes=return_nodes,
            train_return_nodes=train_return_nodes,
            eval_return_nodes=eval_return_nodes,
            tracer_kwargs=tracer_kwargs,
            suppress_diff_warning=suppress_diff_warnings
        )

    def forward(self, x):
        """Forward function of TorchvisionBackbone
        
        Args:
            x (torch.Tensor): the input tensor for feature extraction.
        
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        outs = self.feature_extractor(x)
        return outs