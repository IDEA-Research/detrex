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

import warnings

from detectron2.modeling.backbone import Backbone
from detectron2.utils.logger import setup_logger

try:
    import timm
    from timm.utils.model import freeze_batch_norm_2d
except ImportError:
    timm = None


def log_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.

    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = setup_logger(name="timm backbone")
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
    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=False,
        checkpoint_path="",
        in_channels=3,
        out_indices=(0, 1, 2, 3),
        out_features=("p0", "p1", "p2", "p3"),
        freeze_at=None,
        **kwargs,
    ):
        super().__init__()
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

        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            out_indices=out_indices,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        feature_info = getattr(self.timm_model, "feature_info", None)
        log_timm_feature_info(feature_info)

        output_feature_channels = {"p{}".format(i): feature_info.channels()[i] for i in out_indices}
        out_feature_strides = {"p{}".format(i): feature_info.reduction()[i] for i in out_indices}

        self._out_features = out_features
        self._out_feature_channels = {feat: output_feature_channels[feat] for feat in out_features}
        self._out_feature_strides = {feat: out_feature_strides[feat] for feat in out_features}

        self.out_indices = out_indices
        self.out_features = out_features

    def forward(self, x):
        features = self.timm_model(x)
        outs = {}
        for i in self.out_indices:
            out = features[i]
            outs["p{}".format(i)] = out

        output = {feat: outs[feat] for feat in self.out_features}
        return output
