from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
register_coco_instances("ab_4_cls_train", {
}, "/media/DATADISK/coco_datasets/ab_train/coco_labels.json", "/")
register_coco_instances("ab_4_cls_test",  {
}, "/media/DATADISK/coco_datasets/amba_taiwan_train_0208/coco_labels.json", "/")


# hyper-param for large resolution training and testing
train_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

scale_factor = 1.0
train_scales = [int(scale * scale_factor) for scale in train_scales]  # 1.5
eval_scale = max(train_scales)

max_size = 2000
central_crop_height = 300

# create coco dataset
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ab_4_cls_test"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=train_scales,
                max_size=2000,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            T.CropTransform(0, int(960 - central_crop_height/2),
                            3840, central_crop_height),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=1,  # 16
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="ab_4_cls_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=eval_scale,
                max_size=max_size,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
