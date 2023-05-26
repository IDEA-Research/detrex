import copy
import torch.nn as nn
from easydict import EasyDict

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from projects.co_mot.modeling import (
    COMOT,
    MOTJoiner,
    MOTBackbone,
    MOTPositionEmbeddingSine,
    MOTDeformableTransformer,
    MOTHungarianMatcherGroup,
    MOTQueryInteractionModuleGroup,
    MOTClipMatcher,
    MOTTrackerPostProcess,
    MOTRuntimeTrackerBase,
)
num_frames_per_batch=5
cls_loss_coef=2
bbox_loss_coef=5
giou_loss_coef=2
aux_loss=True
dec_layers=6
g_size=3
weight_dict = {}
for i in range(num_frames_per_batch):
    weight_dict.update({"frame_{}_loss_ce".format(i): cls_loss_coef,
                        'frame_{}_loss_bbox'.format(i): bbox_loss_coef,
                        'frame_{}_loss_giou'.format(i): giou_loss_coef,
                        })
# TODO this is a hack
if aux_loss:
    for i in range(num_frames_per_batch):
        for j in range(dec_layers - 1):
            weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): cls_loss_coef,
                                'frame_{}_aux{}_loss_bbox'.format(i, j): bbox_loss_coef,
                                'frame_{}_aux{}_loss_giou'.format(i, j): giou_loss_coef,
                                })
        for j in range(dec_layers):
            weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): cls_loss_coef,
                                'frame_{}_ps{}_loss_bbox'.format(i, j): bbox_loss_coef,
                                'frame_{}_ps{}_loss_giou'.format(i, j): giou_loss_coef,
                                })

model = L(COMOT)(
    backbone=L(MOTJoiner)(
        backbone = L(MOTBackbone)(
            name = 'resnet50',
            train_backbone = True,
            return_interm_layers = True,
            dilation = False,
        ),
        position_embedding = L(MOTPositionEmbeddingSine)(
            num_pos_feats = 128, 
            normalize=True
        )
    ),
    transformer=L(MOTDeformableTransformer)(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=dec_layers,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=60,
        decoder_self_cross=not False,
        sigmoid_attn=False,
        extra_track_attn=False,
        memory_bank=None,
    ),
    track_embed=L(MOTQueryInteractionModuleGroup)(
        args=EasyDict(random_drop=0.1,
                    fp_ratio=0.3,
                    update_query_pos=False,
                    merger_dropout=0.0,
        ),
        dim_in=256,
        hidden_dim=1024, 
        dim_out=256*2,
    ),
    track_base=L(MOTRuntimeTrackerBase)(score_thresh=0.5, filter_score_thresh=0.5, miss_tolerance=20),
    post_process=L(MOTTrackerPostProcess)(g_size=g_size),
    criterion=L(MOTClipMatcher)(
        num_classes=1,
        matcher=L(MOTHungarianMatcherGroup)(
            cost_class=cls_loss_coef,
            cost_bbox=bbox_loss_coef,
            cost_giou=giou_loss_coef,
        ), 
        weight_dict=weight_dict, 
        losses=['labels', 'boxes'], 
        g_size=g_size
    ),
    num_feature_levels=4,
    num_classes=1,
    num_queries=60,
    aux_loss=True,
    with_box_refine=True,
    two_stage=False,
    memory_bank=None,
    use_checkpoint=True,
    query_denoise=0.0,
    g_size = g_size,
    args=EasyDict(),
)

model.device="cuda"

# # set aux loss weight dict
# base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
# if model.aux_loss:
#     weight_dict = model.criterion.weight_dict
#     aux_weight_dict = {}
#     aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
#     for i in range(model.transformer.decoder.num_layers - 1):
#         aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
#     weight_dict.update(aux_weight_dict)
#     model.criterion.weight_dict = weight_dict
