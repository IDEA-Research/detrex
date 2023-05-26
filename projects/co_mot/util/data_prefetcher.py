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
from functools import partial
from detectron2.structures import Instances

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


def tensor_to_cuda(tensor: torch.Tensor, device):
    return tensor.to(device)


def is_tensor_or_instances(data):
    return isinstance(data, torch.Tensor) or isinstance(data, Instances)


def data_apply(data, check_func, apply_func):
    if isinstance(data, dict):
        for k in data.keys():
            if check_func(data[k]):
                data[k] = apply_func(data[k])
            elif isinstance(data[k], dict) or isinstance(data[k], list):
                data_apply(data[k], check_func, apply_func)
            elif isinstance(data[k], int):
                pass
            else:
                raise ValueError()
    elif isinstance(data, list):
        for i in range(len(data)):
            if check_func(data[i]):
                data[i] = apply_func(data[i])
            elif isinstance(data[i], dict) or isinstance(data[i], list):
                data_apply(data[i], check_func, apply_func)
            elif isinstance(data[i], int):
                pass
            else:
                raise ValueError("invalid type {}".format(type(data[i])))
    elif isinstance(data, int):
        pass
    else:
        raise ValueError("invalid type {}".format(type(data)))
    return data


def data_dict_to_cuda(data_dict, device):
    return data_apply(data_dict, is_tensor_or_instances, partial(tensor_to_cuda, device=device))


class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                print("catch_stop_iter")
                samples = None
                targets = None

        return samples, targets
