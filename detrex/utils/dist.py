# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Utilities for bounding box manipulation and GIoU
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
# ------------------------------------------------------------------------------------------------

import torch.distributed as dist

import os
import torch
import builtins
import datetime
import subprocess

from detectron2.utils import comm


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized() -> bool:
    """
    Checking if the distributed package is available and
    the default process group has been initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """
    Returns the number of processes.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def slurm_init_distributed_mode(args):
    
    assert 'SLURM_PROCID' in os.environ
    assert hasattr(args, 'slurm')
    
    ######################################
    # NOTE: using file://xxxx as dis_url is not stable
    # https://shomy.top/2022/01/05/torch-ddp-intro/
    if args.slurm.ddp_comm_mode == 'tcp':
        node_list = os.environ['SLURM_NODELIST']
        master_addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        
        # explicit tcp url
        args.dist_url = "tcp://%s:%s"%(master_addr, args.slurm.master_port)

        # alternatively, use env vars as below
        # os.environ['MASTER_ADDR'] = master_addr
        # os.environ['MASTER_PORT'] = f'{args.slurm.master_port}'
        # os.environ['RANK'] = str(args.rank)
        # os.environ['LOCAL_RANK'] = str(args.rank % torch.cuda.device_count())
        # os.environ['WORLD_SIZE'] = str(args.world_size)
        # args.dist_url = "env://"
    ######################################

    args.rank = int(os.environ['SLURM_PROCID'])
    args.gpu = args.rank % torch.cuda.device_count()

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    assert comm._LOCAL_PROCESS_GROUP is None
    n_gpus_per_machine = args.slurm.ngpus
    num_machines = args.world_size // n_gpus_per_machine
    machine_rank = args.rank // n_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * n_gpus_per_machine, (i + 1) * n_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg
    comm.synchronize()
    
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
