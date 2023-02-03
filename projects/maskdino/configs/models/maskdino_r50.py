import torch.nn as nn
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L

from projects.maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead
from projects.maskdino.modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from projects.maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
from projects.maskdino.modeling.criterion import SetCriterion
from projects.maskdino.modeling.matcher import HungarianMatcher
from projects.maskdino.maskdino import MaskDINO
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm



dim=256
n_class=80
dn="seg"
dec_layers = 9
input_shape={'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)}
model = L(MaskDINO)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    sem_seg_head=L(MaskDINOHead)(
        input_shape=input_shape,
        num_classes=n_class,
        pixel_decoder=L(MaskDINOEncoder)(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=dim,
            mask_dim=dim,
            norm = 'GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high',
        ),
        loss_weight= 1.0,
        ignore_value= -1,
        transformer_predictor=L(MaskDINODecoder)(
            in_channels=dim,
            mask_classification=True,
            num_classes="${..num_classes}",
            hidden_dim=dim,
            num_queries=300,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            mask_dim=dim,
            enforce_input_project=False,
            two_stage=True,
            dn=dn,
            noise_scale=0.4,
            dn_num=100,
            initialize_box_type='mask2box',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels= "${..pixel_decoder.total_num_feature_levels}",
            dropout = 0.0,
            activation= 'relu',
            nhead= 8,
            dec_n_points= 4,
            return_intermediate_dec = True,
            query_dim= 4,
            dec_layer_share = False,
            semantic_ce_loss = False,
        ),
    ),
    criterion=L(SetCriterion)(
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class = 4.0,
            cost_mask = 5.0,
            cost_dice = 5.0,
            num_points = 12544,
            cost_box=5.0,
            cost_giou=2.0,
            panoptic_on="${..panoptic_on}",
        ),
        weight_dict=dict(),
        eos_coef=0.1,
        losses=['labels', 'masks', 'boxes'],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        dn=dn,
        dn_losses=['labels', 'masks', 'boxes'],
        panoptic_on="${..panoptic_on}",
        semantic_ce_loss=False
    ),
    num_queries=300,
    object_mask_threshold=0.25,
    overlap_threshold=0.8,
    metadata=MetadataCatalog.get('coco_2017_train'),
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    # inference
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
    pano_temp=0.06,
    focus_on_box = False,
    transform_eval = True,
)

# set aux loss weight dict
class_weight=4.0
mask_weight=5.0
dice_weight=5.0
box_weight=5.0
giou_weight=2.0
weight_dict = {"loss_ce": class_weight}
weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})
# two stage is the query selection scheme

interm_weight_dict = {}
interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
weight_dict.update(interm_weight_dict)
# denoising training

if dn == "standard":
    weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
    dn_losses = ["labels", "boxes"]
elif dn == "seg":
    weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
    dn_losses = ["labels", "masks", "boxes"]
else:
    dn_losses = []

aux_weight_dict = {}
for i in range(dec_layers):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
model.criterion.weight_dict=weight_dict