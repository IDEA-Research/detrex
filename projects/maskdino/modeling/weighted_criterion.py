from .criterion import SetCriterion


class WeightedCriterion(SetCriterion):

  def __init__(
    self, 
    num_classes, 
    matcher, 
    weight_dict, 
    eos_coef, 
    losses, 
    num_points, 
    oversample_ratio, 
    importance_sample_ratio, 
    dn="no", 
    dn_losses=..., 
    panoptic_on=False, 
    semantic_ce_loss=False,
    class_weight=4.0,
    mask_weight=5.0,
    dice_weight=5.0,
    box_weight=5.0,
    giou_weight=2.0,
    dec_layers=9):
    # Parse weight dict if it's empty.
    if not isinstance(weight_dict, dict) or len(weight_dict) == 0:
      weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_bbox": box_weight, "loss_giou": giou_weight}
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
      # if deep_supervision
      aux_weight_dict = {}
      for i in range(dec_layers):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
      weight_dict.update(aux_weight_dict)
    
    super().__init__(num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio, dn, dn_losses, panoptic_on, semantic_ce_loss)