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

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["DetrDatasetMapper"]


class DetrDatasetMapper:
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

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
