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

from .detr_dataset_mapper import DetrDatasetMapper
from .mot_dataset_mapper import MotDatasetMapper, MotDatasetInferenceMapper
from .dataset_mappers import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
)
from . import datasets
from .transforms import ColorAugSSDTransform

from .mot_build import build_mot_train_loader, build_mot_test_loader, mot_collate_fn