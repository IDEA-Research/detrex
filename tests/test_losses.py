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

import numpy as np
import unittest
import torch
import torch.nn.functional as F

from detrex.layers.box_ops import generalized_box_iou
from detrex.modeling.losses import CrossEntropyLoss, FocalLoss, GIoULoss, L1Loss

from utils import sigmoid_focal_loss


class TestLosses(unittest.TestCase):
    def test_sigmoid_focal_loss(self):
        num_classes = 80
        preds = torch.randn(2, 300, num_classes)
        targets = torch.randint(size=[2, 300], high=num_classes).long()
        num_boxes = 16
        loss_weight = 2.0

        # detrex focal loss
        focal_loss_detrex = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction="mean",
            loss_weight=loss_weight,
            activated=False,
        )
        detrex_input_preds = preds.view(-1, num_classes)  # (N, num_classes)
        detrex_input_targets = targets.flatten()  # (N, )
        detrex_output = focal_loss_detrex(
            detrex_input_preds, detrex_input_targets, avg_factor=num_boxes
        )

        # original focal loss
        targets_classes_onehot = torch.zeros(
            [preds.shape[0], preds.shape[1], preds.shape[2] + 1],
            dtype=preds.dtype,
            layout=preds.layout,
            device=preds.device,
        )
        targets_classes_onehot.scatter_(2, targets.unsqueeze(-1), 1)
        targets_classes_onehot = targets_classes_onehot[:, :, :-1]
        original_output = (
            sigmoid_focal_loss(preds, targets_classes_onehot, num_boxes=num_boxes) * preds.shape[1]
        )
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

    def test_l1_loss(self):
        preds = torch.tensor([[-1, -1, 1, 1], [-1, -1, 1, 1]], dtype=torch.float32)
        targets = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
        avg_factor = 2
        loss_weight = 2.0

        l1_loss_detrex = L1Loss(reduction="mean", loss_weight=loss_weight)

        detrex_output = l1_loss_detrex(
            preds,
            targets,
            avg_factor=avg_factor,
        )

        original_output = F.l1_loss(preds, targets, reduction="none")
        original_output = original_output.sum() / avg_factor
        original_output *= loss_weight

        self.assertTrue(
            np.allclose(
                detrex_output.cpu().numpy(),
                original_output.cpu().numpy(),
                1e-7,
                1e-7,
            )
        )

    def test_giou_loss(self):
        preds = torch.tensor([[-1, -1, 1, 1], [-1, -1, 1, 1]], dtype=torch.float32)
        targets = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
        avg_factor = 2
        loss_weight = 2.0

        giou_loss_detrex = GIoULoss(eps=1e-6, reduction="mean", loss_weight=loss_weight)

        detrex_output = giou_loss_detrex(preds, targets, avg_factor=avg_factor)
        original_output = 1 - torch.diag(generalized_box_iou(preds, targets))
        original_output = original_output.sum() / avg_factor
        original_output *= loss_weight

        self.assertTrue(
            np.allclose(
                detrex_output.cpu().numpy(),
                original_output.cpu().numpy(),
                1e-7,
                1e-7,
            )
        )
