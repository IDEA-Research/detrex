'''
Author: 颜峰 && bphengyan@163.com
Date: 2023-05-25 10:10:31
LastEditors: 颜峰 && bphengyan@163.com
LastEditTime: 2023-05-26 15:33:44
FilePath: /detrex/configs/common/data/dancetrack_mot.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts


from projects.co_mot.data import MotDatasetMapper, MotDatasetInferenceMapper, build_mot_test_loader, build_mot_train_loader, mot_collate_fn
from projects.co_mot.data.transforms import mot_transforms as TMOT
from projects.co_mot.evaluation import DancetrackEvaluator


dataloader = OmegaConf.create()

dataloader.train = L(build_mot_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dancetrack_train"),
    mapper=L(MotDatasetMapper)(
        augmentation=TMOT.MotCompose([
            TMOT.MotRandomHorizontalFlip(),
            TMOT.MotRandomSelect(
                TMOT.MotRandomResize([608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992], max_size=1536),
                TMOT.MotCompose([
                    TMOT.MotRandomResize([800, 1000, 1200]),
                    TMOT.FixedMotRandomCrop(800, 1200),
                    TMOT.MotRandomResize([608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992], max_size=1536),
                ])
            ),
            TMOT.MOTHSV(),
            TMOT.MotCompose([
                TMOT.MotToTensor(),
                TMOT.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        ]),
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
        sample_mode='random_interval',
        sample_interval=10,
        num_frames_per_batch=5,
    ),
    total_batch_size=16,
    num_workers=4,
    collate_fn=mot_collate_fn,
)

dataloader.test = L(build_mot_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="dancetrack_val", filter_empty=False),
    mapper=L(MotDatasetInferenceMapper)(),
    batch_size=1,
    num_workers=4,
    collate_fn=None,
)

dataloader.evaluator = L(DancetrackEvaluator)(
    dataset_name="${..test.dataset.names}",
)
