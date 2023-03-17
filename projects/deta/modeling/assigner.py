import torch
import torch.nn as nn

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels

from detrex.layers.box_ops import box_iou, box_cxcywh_to_xyxy

def sample_topk_per_gt(pr_inds, gt_inds, iou, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # find topk matches for each gt
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = iou[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:,None].repeat(1, k)

    # filter to as many matches that gt has
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    return pr_inds3, gt_inds3


# modified from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/roi_heads/roi_heads.py#L123
class Stage2Assigner(nn.Module):
    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 400  # number > 91 to filter out later
        self.batch_size_per_image = num_queries
        self.proposal_matcher = Matcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    
    def forward(self, outputs, targets, return_cost_matrix=False):
        # COCO categories are from 1 to 90. They set num_classes=91 and apply sigmoid.

        bs = len(targets)
        indices = []
        ious = []
        for b in range(bs):
            iou, _ = box_iou(
                  box_cxcywh_to_xyxy(targets[b]['boxes']),
                  box_cxcywh_to_xyxy(outputs['init_reference'][b].detach()),
            )
            matched_idxs, matched_labels = self.proposal_matcher(iou)  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.6, 0 ow]
            sampled_idxs, sampled_gt_classes = self._sample_proposals(  # list of sampled proposal_ids, sampled_id -> [0, num_classes)+[bg_label]
                matched_idxs, matched_labels, targets[b]['labels']
            )
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            indices.append((pos_pr_inds, pos_gt_inds))
            ious.append(iou)
        if return_cost_matrix:
            return indices, ious
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)


# modified from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/proposal_generator/rpn.py#L181
class Stage1Assigner(nn.Module):
    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        self.anchor_matcher = Matcher(thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True)

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            anchors = outputs['anchors'][b]
            if len(targets[b]['boxes']) == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=anchors.device),
                                torch.tensor([], dtype=torch.long, device=anchors.device)))
                continue
            iou, _ = box_iou(
                  box_cxcywh_to_xyxy(targets[b]['boxes']),
                  box_cxcywh_to_xyxy(anchors),
            )
            matched_idxs, matched_labels = self.anchor_matcher(iou)  # proposal_id -> highest_iou_gt_id, proposal_id -> [1 if iou > 0.7, 0 if iou < 0.3, -1 ow]
            matched_labels = self._subsample_labels(matched_labels)

            all_pr_inds = torch.arange(len(anchors))
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_ious = iou[pos_gt_inds, pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)
