import torch
import torch.nn.functional as F
# from detrex.modeling.criterion import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from ..losses import  IA_BCE_loss
from .base_criterion import BaseCriterion

class ManyToOneCriterion(BaseCriterion):
    
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, match_number, gamma,alpha,tau):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict)
        # set some attributes
        self.num_classes = num_classes
        # self.losses = cfg.LOSS.
        # self.pos_norm = #cfg.LO
        self.matcher = matcher
        self.pos_norm_type = 'softmax'
        self.weight_dict = weight_dict
        self.match_number = match_number
        #hyper parameters for qal and eqal
        self.gamma = gamma
        self.alpha = alpha   # not the alpha in focal loss!
        self.initialize_weight_table(match_number,tau)
    def initialize_weight_table(self, match_number, tau):
        self.weight_table = torch.zeros(len(match_number), max(match_number))#.cuda()
        for layer, n in enumerate(match_number):
            self.weight_table[layer][:n] = torch.exp(-torch.arange(n) / tau)
            
            
    def _get_local_rank(self, quality, indices):
        bs = len(indices)
        ind_size = [len(i) for i,_ in indices]
        ind_start = 0
        rank_list = []
        for i in range(bs):
            # split quality of one item
            t = quality[ind_start:ind_start+ind_size[i]]
            ind_start += ind_size[i]
            #suppose candidate bag sizes are equal
            if  t.numel() > 0:
                gt_num = int(max(indices[i][1])+1)
                k = ind_size[i] // gt_num
            else:
                gt_num, k = 0, 0  
            t= t.reshape(gt_num, k)
            t_ind = t.sort(dim=-1,descending=True)[1]
            rank = torch.zeros_like(t, dtype=torch.long, device = t.device)
            rank.scatter_(-1, t_ind, torch.arange(k,device=t.device, dtype=torch.long).repeat(gt_num,1))
            rank_list.append(rank.flatten())

        return torch.cat(rank_list, 0)

        
    def get_loss(self, outputs:dict, targets:list, num_boxes:float, layer_spec:int, specify_indices=None):
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs
        # assert (num_boxes  >=0)
        assert (layer_spec >=0)
        losses = {}
        pred_boxes = outputs['pred_boxes']
        src_logits = outputs['pred_logits']

        ##assume 6 layers here
        layer_id = layer_spec
        # feature = outputs['feature']
        device = pred_boxes.device
        if not specify_indices:
            match_n = self.match_number[layer_id] 
            indices = self.matcher(outputs, targets, match_n)
        else:
            match_n = 1 
            indices = specify_indices
        # losses.update(info)
        #do preparation work
        # import ipdb;ipdb.set_trace()
        
        target_boxes = torch.cat([t["boxes"][v[1]] for t,v in zip(targets, indices)], dim=0)
        target_classes_o = torch.cat([t["labels"][v[1]]
                                     for t,v in zip(targets, indices)])#1d class index  sorted by tgt_ind
        
        #compute normalizer for postive
        pos_idx = self._get_src_permutation_idx(indices)
        pos_idx_c = pos_idx + (target_classes_o.cpu(), )
        src_boxes = pred_boxes[pos_idx]
           

        #sum and average

        if not specify_indices: 
            w_prime = self.weight_table[layer_spec].to(device)
        else:
            w_prime = 1
        #define some hyper  parameters here 
        loss_class,loc_weight = IA_BCE_loss(src_logits, pos_idx_c, src_boxes, 
                                        target_boxes, indices, num_boxes, 
                                        self.alpha, self.gamma, 
                                        w_prime,)
      
        losses.update({'loss_class': loss_class})
        
        #compute regression loss
       
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses['loss_bbox'] = (loc_weight* loss_bbox.sum(dim=-1)).sum() / num_boxes        
        
        # cmpute giou loss for regression 
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = (loc_weight * loss_giou).sum() / num_boxes
        
        return losses,indices
        