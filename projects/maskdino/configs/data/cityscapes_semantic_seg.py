from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import CityscapesSemSegEvaluator
from detectron2.data import MetadataCatalog

# from detrex.data import DetrDatasetMapper
# from projects.maskDINO.data.dataset_mappers.coco_instance_lsj_aug_dataset_mapper import COCOInstanceLSJDatasetMapper, build_transform_gen
from detrex.data.dataset_mappers.mask_former_semantic_dataset_mapper import build_transform_gen, MaskFormerSemanticDatasetMapper

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="cityscapes_fine_sem_seg_train"),
    mapper=L(MaskFormerSemanticDatasetMapper)(
        augmentation=L(build_transform_gen)(
            min_size_train=[int(x * 0.1 * 1024) for x in range(5, 21)],
            max_size_train=4096,
            min_size_train_sampling='choice',
            enabled_crop=True,
            crop_params=dict(crop_type='absolute', crop_size=(512, 1024), single_category_max_area=1.0),
            color_aug_ssd=True,
            img_format='RGB',
        ),
        meta=MetadataCatalog.get("cityscapes_fine_sem_seg_train"),
        size_divisibility=-1,
        is_train=True,
        image_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="cityscapes_fine_sem_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=4096,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(CityscapesSemSegEvaluator)(
    dataset_name="${..test.dataset.names}",
)
