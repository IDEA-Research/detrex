from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.config import LazyCall as L

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_detection_train_loader)(
    dataset=LazyCall(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=LazyCall(DatasetMapper)(
        is_train=True,
        augmentations=[
            LazyCall(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            LazyCall(T.RandomFlip)(horizontal=True),
        ],
        image_format="RGB",
        use_instance_mask=False,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = LazyCall(build_detection_test_loader)(
    dataset=LazyCall(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=LazyCall(DatasetMapper)(
        is_train=False,
        augmentations=[
            LazyCall(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = LazyCall(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)