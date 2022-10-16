from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model

# modify model config
# set loss_class to 1.0 brings better results for deformable-detr-box-refinement under lr=1e-4
model.with_box_refine = True
model.criterion.weight_dict = {
    "loss_class": 1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
}
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_with_box_refinement_50ep"