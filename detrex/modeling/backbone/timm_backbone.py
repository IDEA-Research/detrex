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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Support TIMM Backbone
# Modified from:
# https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/timm_backbone.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/backbone.py
# ------------------------------------------------------------------------------------------------

import logging
import warnings
from typing import Tuple
import torch.nn as nn

from detrex.modeling.backbone import Backbone
from detrex.utils.dist import get_rank

try:
    import timm
except ImportError:
    timm = None


def log_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.

    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = logging.getLogger(name="timm backbone")
    if feature_info is None:
        logger.warning("This backbone does not have feature_info")
    elif isinstance(feature_info, list):
        for feat_idx, each_info in enumerate(feature_info):
            logger.info(f"backbone feature_info[{feat_idx}]: {each_info}")
    else:
        try:
            logger.info(f"backbone out_indices: {feature_info.out_indices}")
            logger.info(f"backbone out_channels: {feature_info.channels()}")
            logger.info(f"backbone out_strides: {feature_info.reduction()}")
        except AttributeError:
            logger.warning("Unexpected format of backbone feature_info")


class TimmBackbone(Backbone):
    """A wrapper for using backbone from timm library.

    Please see the document for `feature extraction with timm
    <https://rwightman.github.io/pytorch-image-models/feature_extraction/>`_
    for more details.

    Args:
        model_name (str): Name of timm model to instantiate.
        features_only (bool): Whether to extract feature pyramid (multi-scale
            feature maps from the deepest layer of each stage).
        pretrained (bool): Whether to load pretrained weights. Default: False.
        checkpoint_path (str): Whether to load pretrained weights. Default: False.
        in_channels (int): The number of input channels. Default: 3.
        out_indices (tuple[str]): The extracted feature indices which select
            specific feature levels or limit the stride of the feature extractor.
        out_features (tuple[str]): A map for the output feature dict, e.g.,
            set ("p0", "p1") to return only the feature from indices (0, 1) as
            ``{"p0": feature from indice 0, "p1": feature from indice 1}``.
        norm_layer (nn.Module): Set the specified norm layer for feature extractor,
            e.g., set ``norm_layer=FrozenBatchNorm2d`` to freeze the norm layer
            in feature extractor.
    """

    def __init__(
        self,
        model_name: str,
        features_only: bool = True,
        pretrained: bool = False,
        checkpoint_path: str = "",
        in_channels: int = 3,
        out_indices: Tuple[int] = (0, 1, 2, 3),
        norm_layer: nn.Module = None,
    ):
        super().__init__()
        logger = logging.getLogger(name="timm backbone")
        if timm is None:
            raise RuntimeError('Failed to import timm. Please run "pip install timm". ')
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be bool, not str for model path")
        if features_only and checkpoint_path:
            warnings.warn(
                "Using both features_only and checkpoint_path may cause error"
                " in timm. See "
                "https://github.com/rwightman/pytorch-image-models/issues/488"
            )

        try:
            self.timm_model = timm.create_model(
                model_name=model_name,
                features_only=features_only,
                pretrained=pretrained,
                in_chans=in_channels,
                out_indices=out_indices,
                checkpoint_path=checkpoint_path,
                norm_layer=norm_layer,
            )
        except Exception as error:
            if "feature_info" in str(error):
                raise AttributeError(
                    "Using features_only may cause attribute error"
                    " in timm, cause there's no feature_info attribute in some models. See "
                    "https://github.com/rwightman/pytorch-image-models/issues/1438"
                )
            elif "norm_layer" in str(error):
                raise ValueError(
                    f"{model_name} does not support specified norm layer, please set 'norm_layer=None'"
                )
            else:
                logger.info(error)
                exit()

        self.out_indices = out_indices

        feature_info = getattr(self.timm_model, "feature_info", None)
        if get_rank() == 0:
            log_timm_feature_info(feature_info)

        if feature_info is not None:
            output_feature_channels = {
                "p{}".format(out_indices[i]): feature_info.channels()[i] for i in range(len(out_indices))
            }
            out_feature_strides = {
                "p{}".format(out_indices[i]): feature_info.reduction()[i] for i in range(len(out_indices))
            }

            self._out_features = {"p{}".format(out_indices[i]) for i in range(len(out_indices))}
            self._out_feature_channels = {
                feat: output_feature_channels[feat] for feat in self._out_features
            }
            self._out_feature_strides = {feat: out_feature_strides[feat] for feat in self._out_features}

    def forward(self, x):
        """Forward function of `TimmBackbone`.

        Args:
            x (torch.Tensor): the input tensor for feature extraction.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p1") to tensor
        """
        features = self.timm_model(x)
        outs = {}
        for i in range(len(self.out_indices)):
            out = features[i]
            outs["p{}".format(self.out_indices[i])] = out

        return outs

