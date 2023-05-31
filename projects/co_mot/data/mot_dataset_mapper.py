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
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch
import itertools
from PIL import Image
from typing import Optional
from random import choice, randint
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from detectron2.utils import comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances

__all__ = ["MotDatasetMapper", "MotDatasetInferenceMapper"]


class MotDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        mask_on=False,
        img_format="RGB",
        sample_mode='random_interval',
        sample_interval=10,
        num_frames_per_batch=5,
    ):
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train
        
        self.sample_mode = sample_mode
        assert self.sample_mode == 'random_interval'
        self.sample_interval=sample_interval
        self.num_frames_per_batch=num_frames_per_batch

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape):
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def __call__(self, dataset, cur_idx):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = dataset[cur_idx]
        
        rate = randint(1, self.sample_interval + 1)
        tmax = dataset_dict['t_max'] - dataset_dict['frame_id'] - 1
        indexes = [min(rate * i, tmax) + cur_idx for i in range(self.num_frames_per_batch)]
        
        images, targets = [], []
        for cur_idx in indexes:
            dataset_dict = dataset[cur_idx]
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            assert dataset_dict['boxes_type'] == "x0y0wh"
        
            img = Image.open(dataset_dict["file_name"])
            w, h = img._size
            assert self.img_format == 'RGB' and w == dataset_dict['width'] and h == dataset_dict['height']
            
            images.append(img)
            targets.append(dataset_dict)

        if self.augmentation is not None:
            images, targets = self.augmentation(images, targets)
        
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'width': w,
            'height': h,
        }


import os
import cv2
import torchvision.transforms.functional as TransF
from torch.utils.data import Dataset, DataLoader
class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        if len(self.det_db):
            for line in self.det_db[f_path[:-4].replace('dancetrack/', 'DanceTrack/') + '.txt']:
                l, t, w, h, s = list(map(float, line.split(',')))
                proposals.append([(l + w / 2) / im_w,
                                    (t + h / 2) / im_h,
                                    w / im_w,
                                    h / im_h,
                                    s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5), f_path

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = TransF.normalize(TransF.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):  # 加载图像和proposal。并对图像颜色通道转换+resize+normalize+to_tensor。
        img, proposals, f_path = self.load_img_from_file(self.img_list[index])
        img, ori_img, proposals = self.init_img(img, proposals)
        return img, ori_img, proposals, f_path

class MotDatasetInferenceMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self
    ):
        pass

    def __call__(self, dataset, cur_idx):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = dataset[cur_idx]
        
        img_list = [d['file_name'] for d in dataset_dict]
        img_list = sorted(img_list)
        loader = DataLoader(ListImgDataset('', img_list, ''), 1, num_workers=2)
        
        return {
            'data_loader': loader, 
            "dataset_dict": dataset_dict,
            }
