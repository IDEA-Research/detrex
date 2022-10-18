from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model

# modify model config
# set loss_class to 1.0 brings better results for deformable-detr-box-refinement under lr=1e-4
model.with_box_refine = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_with_box_refinement_50ep"