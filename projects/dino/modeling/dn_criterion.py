import torch


from detrex.utils import (
    get_world_size,
    is_dist_avail_and_initialized,
)

# from ..losses import dice_loss, sigmoid_focal_loss
from .two_stage_criterion import TwoStageCriterion


class DINOCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses=super(DINOCriterion, self).forward(outputs, targets)
        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes,dn_num,single_padding= dn_metas['output_known_lbs_bboxes'], \
                                                       dn_metas['dn_num'], dn_metas['single_padding']
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(dn_num)) * single_padding).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num,
                                            **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux=output_known_lbs_bboxes['aux_outputs'][i]
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes_aux, targets, dn_idx, num_boxes * dn_num,
                                                **kwargs))
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses