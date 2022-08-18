# # ------------------------------------------------------------------------
# # DN-DETR
# # Copyright (c) 2022 IDEA. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
#
#
# import torch
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)
# # from .DABDETR import sigmoid_focal_loss
# from util import box_ops
# import torch.nn.functional as F
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from ideadet.layers import box_ops
# from ideadet.utils import (
#     accuracy,
#     get_world_size,
#     interpolate,
#     is_dist_avail_and_initialized,
#     nested_tensor_from_tensor_list,
# )
#
#
# def prepare_for_dn(targets, dn_args, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
#     """
#     prepare for dn components in forward function
#     Args:
#         dn_args: (targets, args.scalar, args.label_noise_scale,
#                                                              args.box_noise_scale, args.num_patterns) from engine input
#         embedweight: positional queries as anchor
#         training: whether it is training or inference
#         num_queries: number of queries
#         num_classes: number of classes
#         hidden_dim: transformer hidden dimenstion
#         label_enc: label encoding embedding
#
#     Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
#     """
#     scalar, label_noise_scale, box_noise_scale = dn_args
#
#     num_patterns = 1
#     indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
#     tgt = label_enc(torch.tensor(num_classes).cuda()).repeat(num_queries * num_patterns, 1)
#     tgt = torch.cat([tgt, indicator0], dim=1)
#     refpoint_emb = embedweight.repeat(num_patterns, 1)
#     if training:
#         known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
#         know_idx = [torch.nonzero(t) for t in known]
#         known_num = [sum(k) for k in known]
#         # you can uncomment this to use fix number of dn queries
#         # if int(max(known_num))>0:
#         #     scalar=scalar//int(max(known_num))
#
#         # can be modified to selectively denosie some label or boxes; also known label prediction
#         unmask_bbox = unmask_label = torch.cat(known)
#         labels = torch.cat([t['labels'] for t in targets])
#         boxes = torch.cat([t['boxes'] for t in targets])
#         batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
#
#         known_indice = torch.nonzero(unmask_label + unmask_bbox)
#         known_indice = known_indice.view(-1)
#
#         # add noise
#         known_indice = known_indice.repeat(scalar, 1).view(-1)
#         known_labels = labels.repeat(scalar, 1).view(-1)
#         known_bid = batch_idx.repeat(scalar, 1).view(-1)
#         known_bboxs = boxes.repeat(scalar, 1)
#         known_labels_expaned = known_labels.clone()
#         known_bbox_expand = known_bboxs.clone()
#
#         # noise on the label
#         if label_noise_scale > 0:
#             p = torch.rand_like(known_labels_expaned.float())
#             chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
#             new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
#             known_labels_expaned.scatter_(0, chosen_indice, new_label)
#         # noise on the box
#         if box_noise_scale > 0:
#             diff = torch.zeros_like(known_bbox_expand)
#             diff[:, :2] = known_bbox_expand[:, 2:] / 2
#             diff[:, 2:] = known_bbox_expand[:, 2:]
#             known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
#                                            diff).cuda() * box_noise_scale
#             known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
#
#         m = known_labels_expaned.long().to('cuda')
#         input_label_embed = label_enc(m)
#         # add dn part indicator
#         indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
#         input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
#         input_bbox_embed = inverse_sigmoid(known_bbox_expand)
#         single_pad = int(max(known_num))
#         pad_size = int(single_pad * scalar)
#         padding_label = torch.zeros(pad_size, hidden_dim).cuda()
#         padding_bbox = torch.zeros(pad_size, 4).cuda()
#         input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
#         input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
#
#         # map in order
#         map_known_indice = torch.tensor([]).to('cuda')
#         if len(known_num):
#             map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
#             map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
#         if len(known_bid):
#             input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
#             input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
#
#         tgt_size = pad_size + num_queries * num_patterns
#         attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
#         # match query cannot see the reconstruct
#         attn_mask[pad_size:, :pad_size] = True
#         # reconstruct cannot see each other
#         for i in range(scalar):
#             if i == 0:
#                 attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
#             if i == scalar - 1:
#                 attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
#             else:
#                 attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
#                 attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
#         mask_dict = {
#             'known_indice': torch.as_tensor(known_indice).long(),
#             'batch_idx': torch.as_tensor(batch_idx).long(),
#             'map_known_indice': torch.as_tensor(map_known_indice).long(),
#             'known_lbs_bboxes': (known_labels, known_bboxs),
#             'know_idx': know_idx,
#             'pad_size': pad_size
#         }
#     else:  # no dn for inference
#         input_query_label = tgt.repeat(batch_size, 1, 1)
#         input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
#         attn_mask = None
#         mask_dict = None
#
#     input_query_label = input_query_label.transpose(0, 1)
#     input_query_bbox = input_query_bbox.transpose(0, 1)
#
#     return input_query_label, input_query_bbox, attn_mask, mask_dict
#
#
# def dn_post_process(outputs_class, outputs_coord, mask_dict):
#     """
#     post process of dn after output from the transformer
#     put the dn part in the mask_dict
#     """
#     if mask_dict and mask_dict['pad_size'] > 0:
#         output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
#         output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
#         outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
#         outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
#         mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)
#     return outputs_class, outputs_coord
#
#
