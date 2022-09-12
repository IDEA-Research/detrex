import copy
from detrex.config import get_config

from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher

from projects.dino.modeling import (
    DabDeformableDETR,
    DabDeformableDetrTransformerEncoder,
    DabDeformableDetrTransformerDecoder,
    DabDeformableDetrTransformer,
    DINOCriterion
)



from .models.dab_deformable_detr_r50 import model




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



dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
# train.init_checkpoint = "/comp_robot/rentianhe/code/detrex/test_dino.pth"
train.output_dir = "./output/dino_r50_12ep_900query_use_deformable_pos"
# train.output_dir = "./output/test"
train.max_iter = 90000
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.seed = 42

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16
