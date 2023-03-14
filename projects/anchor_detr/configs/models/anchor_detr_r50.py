from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.criterion import SetCriterion
from detrex.modeling.backbone.torchvision_resnet import TorchvisionResNet

from projects.anchor_detr.modeling import (
    AnchorDETR,
    AnchorDETRTransformer,
)


model = L(AnchorDETR)(
    backbone=L(TorchvisionResNet)(
        name="resnet50",
        train_backbone=True,
        dilation=False,
        return_layers={"layer4": "res5"}
    ),
    in_features=["res5"],  # only use last level feature in Conditional-DETR
    in_channels=2048,
    embed_dim=256,
    transformer=L(AnchorDETRTransformer)(
        embed_dim=256, 
        num_heads=8,
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.,
        activation="relu", 
        num_query_position=300,
        num_query_pattern=3,
        spatial_prior="learned",
        attention_type="RCDA",
        num_classes=80,
    ),
    criterion=L(SetCriterion)(
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
            "loss_class": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    aux_loss=True,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=100,
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.num_decoder_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
