'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-25 11:00:08
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-31 10:19:30
FilePath: /detrex/projects/co_mot/data/datasets/register_dancetrack_mot.py
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
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_coco_panoptic_annos_semseg.py
# ------------------------------------------------------------------------------------------------

import json
import os
import logging
import torch
from PIL import Image
from fvcore.common.timer import Timer
from collections import defaultdict

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

DANCETRACK_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
]
def get_dancetrack_mot_instances_meta(dataset_name, seqmap):
    thing_classes = [k["name"][0] for k in DANCETRACK_CATEGORIES]
    meta = {"thing_classes": thing_classes}
    meta['seqmap_txt'] = seqmap
    return meta


def load_dancetrack_mot(image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    def _add_mot_folder(image_root):
        logger.info('YF: Adding {} not exists'.format(image_root))
        labels_full = defaultdict(lambda : defaultdict(list))
        for vid in os.listdir(image_root):
            vid = os.path.join(image_root, vid)
            gt_path = os.path.join(vid, 'gt', 'gt.txt')
            if not os.path.exists(gt_path):
                logger.warning('YF: {} not exists'.format(gt_path))
                continue
            for l in open(gt_path):
                t, i, *xywh, mark, label = l.strip().split(',')[:8]
                t, i, mark, label = map(int, (t, i, mark, label))
                if mark == 0:
                    continue
                if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                    continue
                else:
                    crowd = False
                x, y, w, h = map(float, (xywh))
                labels_full[vid][t].append([x, y, w, h, i, crowd])

        return labels_full

    timer = Timer()
    labels_full = _add_mot_folder(image_root)
    vid_files = list(labels_full.keys())
    
    dataset_dicts = []
    image_id = 0
    obj_idx_offset = 0
    for vid in vid_files:
        t_min = min(labels_full[vid].keys())
        t_max = max(labels_full[vid].keys()) + 1  # 最大帧+1
        obj_idx_offset += 100000  # 100000 unique ids is enough for a video.
        for idx in range(t_min, t_max):
            
            record = {}
            record["file_name"] = os.path.join(image_root, vid, 'img1', f'{idx:08d}.jpg')
            record["not_exhaustive_category_ids"] = []
            record["neg_category_ids"] = []

            record['dataset'] = 'DanceTrack'
            image_id += 1  # imageid必须从1开始
            record['image_id'] = image_id  
            record['frame_id'] = torch.as_tensor(idx)
            record['video_name'] = vid
            record['t_min'] = t_min
            record['t_max'] = t_max
            if idx == t_min:
                img = Image.open(record["file_name"])
                w, h = img._size
            record["height"] = h
            record["width"] = w
            record['size'] = torch.as_tensor([h, w])
            record['orig_size'] = torch.as_tensor([h, w])
            
            record['boxes'] = []
            record['iscrowd'] = []
            record['labels'] = []
            record['obj_ids'] = []
            record['scores'] = []
            record['boxes_type'] = "x0y0wh"
            for *xywh, id, crowd in labels_full[vid][idx]:
                record['boxes'].append(xywh)
                assert not crowd
                record['iscrowd'].append(crowd)
                record['labels'].append(0)
                record['obj_ids'].append(id + obj_idx_offset)
                record['scores'].append(1.)
            record['iscrowd'] = torch.as_tensor(record['iscrowd'])
            record['labels'] = torch.as_tensor(record['labels'])
            record['obj_ids'] = torch.as_tensor(record['obj_ids'], dtype=torch.float64)
            record['scores'] = torch.as_tensor(record['scores'])
            record['boxes'] = torch.as_tensor(record['boxes'], dtype=torch.float32).reshape(-1, 4)
            
            dataset_dicts.append(record)
    
    logger.info("Loading {} takes {:.2f} seconds.".format(image_root, timer.seconds()))
    
    return dataset_dicts


def register_dancetrack_mot_instances(name, metadata, image_root):
    """
    Register a dataset in dancetrack's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_dancetrack_mot(image_root, name))
    MetadataCatalog.get(name).set(image_root=image_root, evaluator_type="mot17", **metadata)



_PREDEFINED_SPLITS_DANCETRACK_MOT = {
    "dancetrack": {
        "dancetrack_train": ("train/", "train_seqmap.txt"),
        "dancetrack_val": ("val/", 'val_seqmap.txt'),
        "dancetrack_test": ("test/", "test_seqmap.txt"),
    },
}


def register_dancetrack_mot(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_DANCETRACK_MOT.items():
        for key, (image_root, seqmap) in splits_per_dataset.items():
            register_dancetrack_mot_instances(
                key,
                get_dancetrack_mot_instances_meta(key, os.path.join(root, seqmap)),
                os.path.join(root, image_root),
            )


_root = os.getenv("DETECTRON2_DATASETS", "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack")
register_dancetrack_mot(_root)


if __name__ == "__main__":
    """
    Test the dataset loader.

    Usage:
        python -m detectron2.data.datasets.lvis \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys
    import numpy as np
    from detectron2.utils.logger import setup_logger
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get('dancetrack_train')

    dicts = load_dancetrack_mot(meta.image_root, meta.name)
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "tmp"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
