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

import unittest
import torch
import torch.nn.functional as F
import numpy as np

from detrex.modeling.losses import (
    FocalLoss,
    CrossEntropyLoss,
)

from utils import sigmoid_focal_loss


class TestLosses(unittest.TestCase):
    def test_sigmoid_focal_loss(self):
        preds = torch.randn(2, 3)
        targets = torch.ones(2, 3)
        num_boxes = 3
        loss_weight=2.0

        focal_loss_detrex = FocalLoss(
            alpha=0.25, 
            gamma=2.0, 
            reduction="mean", 
            loss_weight=loss_weight
        )
        
        detrex_output = focal_loss_detrex(preds, targets, avg_factor=num_boxes)
        original_output = sigmoid_focal_loss(preds, targets, num_boxes=num_boxes) * 3
        original_output *= loss_weight

        self.assertTrue(
            np.allclose(
                detrex_output.cpu().numpy(), 
                original_output.cpu().numpy(),
                1e-7,
                1e-7,
            )
        )
    
    def test_cross_entropy(self):
        num_classes = 81
        empty_weight = torch.ones(num_classes)
        empty_weight[-1] = 0.1
        loss_weight = 2.0
        
        preds = torch.randn(3, num_classes)
        targets = torch.empty(3, dtype=torch.long).random_(num_classes)

        ce_detrex = CrossEntropyLoss(
            reduction="mean",
            loss_weight=loss_weight,
        )

        detrex_output = ce_detrex(preds, targets, class_weight=empty_weight)
        original_output = F.cross_entropy(preds, targets, empty_weight) * loss_weight
        
        self.assertTrue(
            np.allclose(
                detrex_output.cpu().numpy(), 
                original_output.cpu().numpy(),
                1e-7,
                1e-7,
            )
        )