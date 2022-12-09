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
# ------------------------------------------------------------------------------------------------
# COCO Instance Segmentation with LSJ Augmentation
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch

from detectron2.structures import Instances, Boxes, PolygonMasks
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(
    image_size,
    min_scale,
    max_scale,
    random_flip: str = "horizontal",
    is_train: bool = True,
):
    """
    Create a list of default :class:`Augmentation`.
    Now it includes resizing and flipping.
    
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation."
    assert random_flip in ["none", "horizontal", "vertical"], f"Only support none/horizontal/vertical flip, but got {random_flip}"

    augmentation = []

    if random_flip != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=random_flip == "horizontal",
                vertical=random_flip == "vertical",
            )
        )
    
    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size,
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size))
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    def __init__(
        self,
        is_train=True,
        *,
        augmentation,
        image_format,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
        )

        self.img_format = image_format
        self.is_train = is_train
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'): 
                instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            dataset_dict["instances"] = instances

        return dataset_dict

