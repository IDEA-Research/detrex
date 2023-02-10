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

import copy
import random
from omegaconf.listconfig import ListConfig

from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation


class AutoAugment(object):
    def __init__(self, policies):
        assert isinstance(policies, (list, ListConfig)) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, (list, ListConfig)) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, Augmentation), \
                    "Each specific augmentation should be the class of \
                        detectron2.data.transforms.Augmentation"
        
        self.policies = copy.deepcopy(policies)


    def __call__(self, inputs):
        augmentation = random.choice(self.policies)
        
        # this is an hack to avoid using omegaconf.listconfig.ListConfig
        augmentation = list(augmentation)
        augmentation = T.AugmentationList(augmentation)

        return augmentation(inputs)


    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'