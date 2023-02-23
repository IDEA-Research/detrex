#!/usr/bin/env python
"""
A script to launch training, it surpports:

* one-line command to launch training locally or on a slurm cluster
* automatic experiment name generation according to hyperparameter overrides
* automatic requeueing & resume from latest checkpoint when a job reaches maximum running time or is preempted

Example usage:

$ python tools/hydra_train_net.py \
     num_machines=2 num_gpus=8 auto_output_dir=true \
     config_file=projects/detr/configs/detr_r50_300ep.py \
     +model.num_queries=50 \
     +slurm=${CLUSTER_ID}

$ tree -L 2 ./outputs/
./outputs/
└── +model.num_queries.50-num_gpus.8-num_machines.2
    └── 20230224-09:06:28

Contact ZHU Lei (ray.leizhu@outlook.com) for inquries about this script
"""
import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# print(sys.path)

# FIXME: it seems that, even though I put tools/.. in front of PYTHONPATH, the interpreter still finally 
# find detectron2/tools/train_net.  Two workarounds:
# 1. pip uninstall detrex && pip uninstall detectron2 && pip install detrex && pip install detectron2 (tested)
# 2. PYTHONPATH=${PWD}:${PYTHONPATH} python tools/hydra_train_net.py ... (not tested)
from tools.train_net import main

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

from detectron2.engine import launch
from detectron2.config import LazyConfig
import os.path as osp

import submitit
import uuid
from pathlib import Path

from detrex.utils.dist import slurm_init_distributed_mode


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_shared_folder(share_root) -> Path:
    if Path(share_root).parent.is_dir():
        p = Path(f"{share_root}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError(f"The parent of share_root ({share_root}) must exist!")

def get_init_file(share_root):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(share_root)), exist_ok=True)
    init_file = get_shared_folder(share_root) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def get_dist_url(ddp_comm_mode='tcp', share_root=None):
    if ddp_comm_mode == 'file':
        assert share_root is not None
        return get_init_file(share_root).as_uri()
    elif ddp_comm_mode == 'tcp':
        return 'env://'
    else:
        raise ValueError('Unknown DDP communication mode')

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        if self.args.world_size > 1:
            slurm_init_distributed_mode(self.args)
        if not self.args.eval_only: # always auto resume if in training
            self.args.resume = True 
        main(self.args)

    def checkpoint(self): # being called when met timeout or preemption signal is received
        import os
        import submitit

        self.args.dist_url = get_dist_url(
            ddp_comm_mode=self.args.slurm.ddp_comm_mode,
            share_root=self.args.slurm.share_root)

        self.args.resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        # https://shomy.top/2022/01/05/torch-ddp-intro/
        # self.args.dist_url = f'tcp://{job_env.hostname}:{self.args.slurm.port}'
        # self.args.output_dir = self.args.slurm.job_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        self.args.machine_rank = job_env.node

        self.args.slurm.jobid = job_env.job_id # just in case of need, e.g. logging to wandb

        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


# @hydra.main(version_base=None, config_path="../configs/hydra", config_name="train_args.yaml")
@hydra.main(config_path="../configs/hydra", config_name="train_args.yaml")
def hydra_app(args:DictConfig):
    # NOTE: enable write to unknow field of cfg
    # hence it behaves like argparse.NameSpace
    # this is required as some args are determined at runtime
    # https://stackoverflow.com/a/66296809
    OmegaConf.set_struct(args, False)

    # TODO: switch to hydra 1.3+, which natrually supports relative path
    # the following workaround is for hydra 1.1.2
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # since hydra 1.1.2 will change PWD to run dir, get current work dir first
    args.config_file = osp.join(get_original_cwd(), args.config_file)
    
    # command line args starting with '+' are for overrides, except '+slurm=[cluster_id]'
    args.opts = [ x.replace('+', '') for x in hydra_cfg['overrides']['task'] if (x.startswith('+') 
                  and not x.startswith('+slurm'))]
    # print(args.opts)

    hydra_run_dir = os.path.join(get_original_cwd(), hydra_cfg['run']['dir'])
    if args.auto_output_dir:
        args.opts.append(f"train.output_dir={hydra_run_dir}")
    # print(args.opts)

    # test args
    # print(OmegaConf.to_yaml(args, resolve=True))
   
    if not hasattr(args, 'slurm'): # run locally
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    else: # run with slurm
        if args.slurm.job_dir is None: # use hydra run_dir as slurm output dir
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.slurm.job_dir = hydra_run_dir
        
        if args.slurm.master_port is None: # automatically find free port for ddp communication
            args.slurm.master_port = _find_free_port()
        
        executor = submitit.AutoExecutor(folder=args.slurm.job_dir, slurm_max_num_timeout=30)
        
        ############## NOTE: this part is highly dependent on slurm version ##############
        kwargs = {}
        if args.slurm.comment:
            kwargs['slurm_comment'] = args.slurm.comment

        # NOTE: slurm of different versions may have different flags
        # slurm_additional_parameters is flexible to cope with this scenario
        slurm_additional_parameters={'ntasks': args.slurm.nodes*args.slurm.ngpus, 
                                     'gres': f'gpu:{args.slurm.ngpus}',
                                     'ntasks-per-node': args.slurm.ngpus} # one task per GPU
        if args.slurm.exclude_node:
            slurm_additional_parameters['exclude'] = args.slurm.exclude_node
        
        if args.slurm.quotatype:
            slurm_additional_parameters['quotatype'] = args.slurm.quotatype
        ##################################################################################

        executor.update_parameters(
            ## original
            # mem_gb=40 * num_gpus_per_node,
            # gpus_per_node=num_gpus_per_node, 
            # tasks_per_node=num_gpus_per_node,  # one task per GPU
            # nodes=nodes,
            # timeout_min=timeout_min,  # max is 60 * 72
            ## https://github.com/facebookincubator/submitit/issues/1639
            # mem_per_cpu=4000,
            # gpus_per_node=num_gpus_per_node,
            # cpus_per_task=4,
            cpus_per_task=args.slurm.cpus_per_task,
            nodes=args.slurm.nodes,
            slurm_additional_parameters=slurm_additional_parameters,
            timeout_min=args.slurm.timeout * 60, # in minutes
            # Below are cluster dependent parameters
            slurm_partition=args.slurm.partition,
            slurm_signal_delay_s=120,
            **kwargs
        )

        executor.update_parameters(name=args.slurm.job_name)

        args.dist_url = get_dist_url(
            ddp_comm_mode=args.slurm.ddp_comm_mode,
            share_root=args.slurm.share_root)

        trainer = Trainer(args)
        job = executor.submit(trainer)
        
        print("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    hydra_app()
    