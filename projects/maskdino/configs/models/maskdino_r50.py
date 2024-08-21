import torch.nn as nn
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm

from omegaconf import OmegaConf

from ...modeling.meta_arch.maskdino_head import MaskDINOHead
from ...modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from ...modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
from ...modeling.weighted_criterion import WeightedCriterion
from ...modeling.matcher import HungarianMatcher
from ...maskdino import MaskDINO



model = L(MaskDINO)(
    # parameters in one place.
    params=OmegaConf.create(dict(
        input_shape={
            'res2': L(ShapeSpec)(channels=256, height=None, width=None, stride=4), 
            'res3': L(ShapeSpec)(channels=512, height=None, width=None, stride=8), 
            'res4': L(ShapeSpec)(channels=1024, height=None, width=None, stride=16), 
            'res5': L(ShapeSpec)(channels=2048, height=None, width=None, stride=32)
        },
        dim=256,
        hidden_dim=256,
        query_dim=4,
        num_classes=80,
        dec_layers=9,
        enc_layers=6,
        feed_forward=2048,
        n_heads=8,
        num_queries=300,
        dn_num=100,
        dn_mode="seg",
        show_weights=True
    )),
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
        input_shape="${..params.input_shape}",
        num_classes="${..params.num_classes}",
        pixel_decoder=L(MaskDINOEncoder)(
            input_shape="${...params.input_shape}",
            transformer_dropout=0.0,
            transformer_nheads="${...params.n_heads}",
            transformer_dim_feedforward="${...params.feed_forward}",
            transformer_enc_layers="${...params.enc_layers}",
            conv_dim="${...params.dim}",
            mask_dim="${...params.dim}",
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
            in_channels="${...params.dim}",
            mask_classification=True,
            num_classes="${...params.num_classes}",
            hidden_dim="${...params.hidden_dim}",
            num_queries="${...params.num_queries}",
            nheads="${...params.n_heads}",
            dim_feedforward="${...params.feed_forward}",
            dec_layers="${...params.dec_layers}",
            mask_dim="${...params.dim}",
            enforce_input_project=False,
            two_stage=True,
            dn="${...params.dn_mode}",
            noise_scale=0.4,
            dn_num="${...params.dn_num}",
            initialize_box_type='mask2box',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels= "${..pixel_decoder.total_num_feature_levels}",
            dropout = 0.0,
            activation= 'relu',
            nhead= "${...params.n_heads}",
            dec_n_points= 4,
            return_intermediate_dec = True,
            query_dim= "${...params.query_dim}",
            dec_layer_share = False,
            semantic_ce_loss = False,
        ),
    ),
    criterion=L(WeightedCriterion)(
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
        # Params for aux loss weight
        class_weight=4.0,
        mask_weight=5.0,
        dice_weight=5.0,
        box_weight=5.0,
        giou_weight=2.0,
        dec_layers="${..params.dec_layers}",
        # Default mask dino options for set criterion.
        weight_dict=dict(),
        eos_coef=0.1,
        losses=['labels', 'masks', 'boxes'],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        dn="${..params.dn_mode}",
        dn_losses=['labels', 'masks', 'boxes'],
        panoptic_on="${..panoptic_on}",
        semantic_ce_loss=False
    ),
    num_queries="${.params.num_queries}",
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
# class_weight=4.0
# mask_weight=5.0
# dice_weight=5.0
# box_weight=5.0
# giou_weight=2.0
# weight_dict = {"loss_ce": class_weight}
# weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
# weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})
# # two stage is the query selection scheme

# interm_weight_dict = {}
# interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
# weight_dict.update(interm_weight_dict)
# # denoising training

# if dn == "standard":
#     weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
#     dn_losses = ["labels", "boxes"]
# elif dn == "seg":
#     weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
#     dn_losses = ["labels", "masks", "boxes"]
# else:
#     dn_losses = []
# # if deep_supervision:

# aux_weight_dict = {}
# for i in range(dec_layers):
#     aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
# weight_dict.update(aux_weight_dict)
# Old way to do it.
# model.criterion.weight_dict=weight_dict