import copy
from detrex.config import get_config
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.criterion.two_stage_criterion import TwoStageCriterion

from detectron2.config import LazyCall as L

from .models.dab_deformable_detr_r50 import model
from .common.coco_loader import dataloader
from .common.schedule import lr_multiplier_50x as lr_multiplier
from modeling.dn_criterion import SetCriterion as DINOCriterion


# set model
model.as_two_stage = True
model.criterion = L(DINOCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn":1,
            'loss_bbox_dn':5.0,
            'loss_giou_dn':2.0
        },

        losses=[
            "class",
            "boxes",
        ],
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    )

model.dn_number = 100
model.label_noise_ratio = 0.2
model.box_noise_scale = 1.0
# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
if model.as_two_stage:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + f"_enc": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# train.init_checkpoint = "/comp_robot/rentianhe/code/IDEADet/test_dab_deformable.pth"
train.output_dir = "./test"
train.max_iter = 375000


optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 16
