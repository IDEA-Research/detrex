# Copyright (c) IDEA, Inc. and its affiliates.
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskdino_decoder import build_transformer_decoder
from ..pixel_decoder.maskdino_encoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskDINOHead(nn.Module):

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

    def forward(self, features, mask=None,targets=None):
        return self.layers(features, mask,targets=targets)

    def layers(self, features, mask=None,targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)

        predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets)

        return predictions
