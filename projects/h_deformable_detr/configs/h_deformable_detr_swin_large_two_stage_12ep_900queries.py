from .h_deformable_detr_r50_50ep import train, dataloader, optimizer, model
from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
# modify model config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    drop_path_rate=0.2,
    window_size=7,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]
model.with_box_refine = True
model.as_two_stage = True

model.num_queries_one2one = 900
model.transformer.two_stage_num_proposals = 2400

# modify training config
train.init_checkpoint = "/mnt/pretrained_backbone/swin_large_patch4_window7_224_22k.pth"
train.output_dir = "./output/h_deformable_detr_swin_large_two_stage_12ep_900queries"
train.max_iter = 90000
