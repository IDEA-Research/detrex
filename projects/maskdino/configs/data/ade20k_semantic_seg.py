from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import SemSegEvaluator
from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
# from detrex.data import DetrDatasetMapper
# from projects.maskDINO.data.dataset_mappers.coco_instance_lsj_aug_dataset_mapper import COCOInstanceLSJDatasetMapper, build_transform_gen
from detrex.data.dataset_mappers import MaskFormerSemanticDatasetMapper,maskformer_semantic_transform_gen

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_sem_seg_train"),
    mapper=L(MaskFormerSemanticDatasetMapper)(
        augmentation=L(maskformer_semantic_transform_gen)(
            min_size_train=[int(x * 0.1 * 512) for x in range(5, 21)],
            max_size_train=2048,
            min_size_train_sampling="choice",
            enabled_crop=True,
            crop_params=dict(
              crop_type="absolute",
              crop_size=(512,512),
              single_category_max_area=1.0,
            ),
            color_aug_ssd=True,
            img_format="RGB",
        ),
        meta=MetadataCatalog.get("ade20k_sem_seg_train"),
        is_train=True,
        image_format="RGB",
        size_divisibility=512,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=512,
                max_size=2048,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(SemSegEvaluator)(
    dataset_name="${..test.dataset.names}",
)
