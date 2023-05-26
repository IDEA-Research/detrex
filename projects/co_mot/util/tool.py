# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import numpy as np
import collections

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(f'loaded {model_path}')
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                if 'class_embed' in k:
                    print("load class_embed: {} shape={}".format(k, state_dict[k].shape))
                    if model_state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        state_dict[k] = state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        state_dict[k] = state_dict[k][1:4]
                    elif model_state_dict[k].shape[0] == 11:
                        state_dict[k] = state_dict[k][1:12]
                    elif model_state_dict[k].shape[0] == 100:
                        state_dict[k] = state_dict[k].repeat_interleave(model_state_dict[k].shape[0]//state_dict[k].shape[0]+1, dim=0)[:model_state_dict[k].shape[0]]
                    elif model_state_dict[k].shape[0] == 91 and state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k].repeat_interleave(91, dim=0)
                    elif model_state_dict[k].shape[0] == 2000:
                        state_dict[k] = state_dict[k].repeat_interleave(model_state_dict[k].shape[0]//state_dict[k].shape[0]+1, dim=0)[:model_state_dict[k].shape[0]]
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                    continue
                state_dict[k] = model_state_dict[k]
        elif k.replace('in_proj_weight', 'in_proj.weight') in model_state_dict:
            # state_dict[k] = model_state_dict[k]
            k_dst = k.replace('in_proj_weight', 'in_proj.weight')
            state_dict = collections.OrderedDict([(k_dst, v) if k_ == k else (k_, v) for k_, v in state_dict.items()])
        elif k.replace('in_proj_bias', 'in_proj.bias') in model_state_dict:
            # state_dict[k] = model_state_dict[k]
            k_dst = k.replace('in_proj_bias', 'in_proj.bias')
            state_dict = collections.OrderedDict([(k_dst, v) if k_ == k else (k_, v) for k_, v in state_dict.items()])
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):  # pretrain model
            # if 'class_embed_two.' in k:
            #     state_dict[k] = state_dict[k.replace('class_embed_two', 'class_embed')]
            # elif 'bbox_embed_two.' in k:
            #     state_dict[k] = state_dict[k.replace('bbox_embed_two', 'bbox_embed')]
            # elif 'decoder.0.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.0.', 'transformer.decoder.')]
            # elif 'decoder.1.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.1.', 'transformer.decoder.')]
            # elif 'decoder.2.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.2.', 'transformer.decoder.')]
            # elif 'decoder.3.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.3.', 'transformer.decoder.')]
            # elif 'decoder.4.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.4.', 'transformer.decoder.')]
            # elif 'decoder.5.' in k:
            #     state_dict[k] = state_dict[k.replace('decoder.5.', 'transformer.decoder.')]
            # if 'bbox_embed_trk.' in k:
            #     state_dict[k] = state_dict[k.replace('bbox_embed_trk', 'bbox_embed')]
            # elif 'class_embed_trk.' in k:
            #     state_dict[k] = state_dict[k.replace('class_embed_trk', 'class_embed')]
            # else:
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model



