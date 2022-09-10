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

import torch
import torch.nn as nn


class GenerateNoiseQueries(nn.Module):
    def __init__(
        self,
        noise_nums: int = 5,
        label_noise_scale: float = 0.0,
        box_noise_scale: float = 0.0,
    ):
        self.noise_nums = noise_nums
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
    
    def apply_label_noise(self, gt_labels_list):
        noised_label = gt_labels_list.repeat(self.noise_nums, 1).view(-1)
        if self.label_noise_scale > 0:
            