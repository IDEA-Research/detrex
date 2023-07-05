<!--
 * @Author: 颜峰 && bphengyan@163.com
 * @Date: 2023-05-26 15:47:34
 * @LastEditors: 颜峰 && bphengyan@163.com
 * @LastEditTime: 2023-05-26 15:54:06
 * @FilePath: /detrex/projects/co_mot/README.md
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
-->
# CO-MOT: Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=bridging-the-gap-between-end-to-end-and-non)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-bdd100k)](https://paperswithcode.com/sota/multi-object-tracking-on-bdd100k?p=bridging-the-gap-between-end-to-end-and-non)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-end-to-end-and-non/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bridging-the-gap-between-end-to-end-and-non)


This repository is an official implementation of [CO-MOT](https://arxiv.org/abs/2305.12724).

**TO DO**
1. release bdd100K, MOT17 model.
2. add DINO backbone

## Introduction

Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking.

**Abstract.** Existing end-to-end Multi-Object Tracking (e2e-MOT) methods have not surpassed non-end-to-end tracking-by-detection methods. One potential reason is its label assignment strategy during training that consistently binds the tracked objects with tracking queries and then assigns the few newborns to detection queries. With one-to-one bipartite matching, such an assignment will yield unbalanced training, i.e., scarce positive samples for detection queries, especially for an enclosed scene, as the majority of the newborns come on stage at the beginning of videos. Thus, e2e-MOT will be easier to yield a tracking terminal without renewal or re-initialization, compared to other tracking-by-detection methods. To alleviate this problem, we present Co-MOT, a simple and effective method to facilitate e2e-MOT by a novel coopetition label assignment with a shadow concept. Specifically, we add tracked objects to the matching targets for detection queries when performing the label assignment for training the intermediate decoders. For query initialization, we expand each query by a set of shadow counterparts with limited disturbance to itself. With extensive ablations, Co-MOT achieves superior performance without extra costs, e.g., 69.4% HOTA on DanceTrack and 52.8% TETA on BDD100K. Impressively, Co-MOT only requires 38\% FLOPs of MOTRv2 to attain a similar performance, resulting in the 1.4× faster inference speed.


## Main Results

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   69.9   |   82.1   |   58.9   |   91.2   |   71.9   | [model](https://drive.google.com/file/d/15HOnAUlYRjFBQVIsek1Qbgf18Pkffy-A/view?usp=share_link) |



## Usage

### Dataset preparation

1. Please download [DanceTrack](https://dancetrack.github.io/) and [CrowdHuman](https://www.crowdhuman.org/) and unzip them as follows:

```
/data/Dataset/mot
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_trainval.odgt
│   ├── annotation_val.odgt
│   └── Images
├── DanceTrack
│   ├── test
│   ├── train
│   └── val
```


## Evaluation
Model evaluation can be done as follows:
```bash
python tools/train_net.py --config-file projects/co_mot/configs/mot_r50_4scale_10ep.py --eval-only train.init_checkpoint=./co_mot_dancetrack.pth train.device=cuda
```

## Demo 
Demo can be done as follows:
```bash
python tools/train_net.py --config-file projects/co_mot/configs/mot_r50.py --video-input ./demo_video.avi  --output visualize_video_results.mp4 --opts train.init_checkpoint=./co_mot_dancetrack.pth train.device=cuda
```

## Citing DINO
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTex
@article{yan2023bridging,
 title={Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking},
 author={Yan, Feng and Luo, Weixin and Zhong, Yujie and Gan, Yiyang and Ma, Lin},
 journal={arXiv preprint arXiv:2305.12724},
 year={2023}
}
```


## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [BDD100K](https://github.com/bdd100k/bdd100k)
- [MOTRv2](https://github.com/megvii-research/MOTRv2)