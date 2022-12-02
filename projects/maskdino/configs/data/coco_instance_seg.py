from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

# from detrex.data import DetrDatasetMapper
# from projects.maskDINO.data.dataset_mappers.coco_instance_lsj_aug_dataset_mapper import COCOInstanceLSJDatasetMapper, build_transform_gen
from detrex.data.dataset_mappers import COCOInstanceNewBaselineDatasetMapper,coco_instance_transform_gen
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(COCOInstanceNewBaselineDatasetMapper)(
        augmentation=L(coco_instance_transform_gen)(
            image_size=1024,
            min_scale=0.1,
            max_scale=2.0,
            random_flip="horizontal"
        ),
        is_train=True,
        image_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(COCOInstanceNewBaselineDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
