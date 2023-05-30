'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-26 10:06:20
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-30 16:03:02
FilePath: /detrex/projects/co_mot/modeling/__init__.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
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


from .mot import MOT
from .mot import ClipMatcher as MOTClipMatcher
from .mot import TrackerPostProcess as MOTTrackerPostProcess
from .mot import RuntimeTrackerBase as MOTRuntimeTrackerBase

from .mot_transformer import DeformableTransformer as MOTDeformableTransformer

from .qim import QueryInteractionModuleGroup as MOTQueryInteractionModuleGroup

from .matcher import HungarianMatcherGroup as MOTHungarianMatcherGroup

