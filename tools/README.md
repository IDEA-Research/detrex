This directory contains a few useful example scripts for detrex.

### train_net.py

A simpler training engine modified from [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py).


### hydra_train_net.py

A script to launch training, it surpports:

* one-line command to launch training locally or on a slurm cluster
* automatic experiment name generation according to hyperparameter overrides
* automatic requeueing & resume from latest checkpoint when a job reaches maximum running time or is preempted

Usage:

* STER 1: modify slurm config
```bash
$ cp configs/hydra/slurm/research.yaml configs/hydra/slurm/${CLUSTER_ID}.yaml && \
    vim configs/hydra/slurm/${CLUSTER_ID}.yaml 
```

* STEP 2: launch training
```bash
$ python tools/hydra_train_net.py \
     num_machines=2 num_gpus=8 auto_output_dir=true \
     config_file=projects/detr/configs/detr_r50_300ep.py \
     +model.num_queries=50 \
     +slurm=${CLUSTER_ID}
```

* STEP 3 (optional): check output dir
```bash
$ tree -L 2 ./outputs/
./outputs/
└── +model.num_queries.50-num_gpus.8-num_machines.2
    └── 20230224-09:06:28
```

**Notes:** 

1. to override hyperparameters specified in lazy config file, use `+[KEY]=[VAL]`
2. to launch training on a slurm cluster, specify a slurm config in `configs/hydra/slurm/[CLUSTER_ID].yaml` first, and then use `+slurm=[CLUSTER_ID]` in command line. Remove `+slurm=[CLUSTER_ID]` to launch locally. 
3. to atomatically generate experiment directory, use `auto_output_dir=true`

### analyze_model.py

Analyze FLOPs, parameters, activations of the detrex model modified from detectron2 [analyze_model.py](https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py).


### benchmark.py

Benchmark the training speed, inference speed or data loading speed of a given config, modified from [benchmark.py](https://github.com/facebookresearch/detectron2/blob/main/tools/benchmark.py)

Usage:
```bash
python tools/benchmark.py --config-file /path/to/config.py --task train/eval/data
```

### visualize_json_results.py

Visualize the json instance detection/segmentation results dumped by `COCOEvaluator` modified from [visualize_json_results.py](https://github.com/facebookresearch/detectron2/blob/main/tools/visualize_json_results.py)

Usage:
```bash
python tools/visualize_json_results.py --input x.json \
                                       --output dir/ \
                                       --dataset coco_2017_val
```
If not using a builtin dataset, you'll need your own script or modify this script.

### visualize_data.py

Visualize ground truth raw annotations or training data (after preprocessing/augmentations) modified from [visualize_data.py](https://github.com/facebookresearch/detectron2/blob/main/tools/visualize_data.py)

Usage:
```bash
python tools/visualize_data.py --config-file /path/to/config.py \
                               --source annotation/dataloader \
                               --output-dir dir/ \
                               [--show]
```

**Notes**: the script does not stop by itself when using `--source dataloader` because a training dataloader is usually infinite.
