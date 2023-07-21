<div align="center">
  <img src="./assets/logo_2.png" width="30%">
</div>
<h2 align="center">🦖detrex: Benchmarking Detection Transformers</h2>
<p align="center">
    <a href="https://github.com/IDEA-Research/detrex/releases">
        <img alt="release" src="https://img.shields.io/github/v/release/IDEA-Research/detrex">
    </a>
    <a href="https://detrex.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href='https://detrex.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/detrex/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/IDEA-Research/detrex/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/IDEA-Research/detrex.svg?color=blue">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/IDEA-Research/detrex">
    </a>
</p>


<div align="center">

<!-- <a href="https://arxiv.org/abs/2306.07265">📚Read detrex Benchmarking Paper</a> <sup><i><font size="3" color="#FF0000">New</font></i></sup> |
<a href="https://rentainhe.github.io/projects/detrex/">🏠Project Page</a> <sup><i><font size="3" color="#FF0000">New</font></i></sup> |  [🏷️Cite detrex](#citation) -->

[📚Read detrex Benchmarking Paper](https://arxiv.org/abs/2306.07265) | [🏠Project Page](https://rentainhe.github.io/projects/detrex/) | [🏷️Cite detrex](#citation) | [🚢DeepDataSpace](https://github.com/IDEA-Research/deepdataspace)

</div>


<div align="center">

[📘Documentation](https://detrex.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) |
[👀Model Zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) |
[🚀Awesome DETR](https://github.com/IDEA-Research/awesome-detection-transformer) |
[🆕News](#whats-new) |
[🤔Reporting Issues](https://github.com/IDEA-Research/detrex/issues/new/choose)

</div>


## Introduction

detrex is an open-source toolbox that provides state-of-the-art Transformer-based detection algorithms. It is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and its module design is partially borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection) and [DETR](https://github.com/facebookresearch/detr). Many thanks for their nicely organized code. The main branch works with **Pytorch 1.10+** or higher (we recommend **Pytorch 1.12**).

<div align="center">
  <img src="./assets/detr_arch.png" width="100%"/>
</div>

<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decomposes the Transformer-based detection framework into various components which help users easily build their own customized models.

- **Strong Baselines.** detrex provides a series of strong baselines for Transformer-based detection models. We have further boosted the model performance from **0.2 AP** to **1.1 AP** through optimizing hyper-parameters among most of the supported algorithms.

- **Easy to Use.** detrex is designed to be **light-weight** and easy for users to use:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight [training engine](./tools/train_net.py) modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)

Apart from detrex, we also released a repo [Awesome Detection Transformer](https://github.com/IDEA-Research/awesome-detection-transformer) to present papers about Transformer for detection and segmentation.

</details>

## Fun Facts
The repo name detrex has several interpretations:
- <font color=blue> <b> detr-ex </b> </font>: We take our hats off to DETR and regard this repo as an extension of Transformer-based detection algorithms.

- <font color=#db7093> <b> det-rex </b> </font>: rex literally means 'king' in Latin. We hope this repo can help advance the state of the art on object detection by providing the best Transformer-based detection algorithms from the research community.

- <font color=#008000> <b> de-t.rex </b> </font>: de means 'the' in Dutch. T.rex, also called Tyrannosaurus Rex, means 'king of the tyrant lizards' and connects to our research work 'DINO', which is short for Dinosaur.

## What's New
v0.5.0 was released on 16/07/2023:
- Support [Focus-DETR (ICCV'2023)](./projects/focus_detr/).
- Support [SQR-DETR (CVPR'2023)](https://github.com/IDEA-Research/detrex/tree/main/projects/sqr_detr), credits to [Fangyi Chen](https://github.com/Fangyi-Chen)
- Support [Align-DETR (ArXiv'2023)](./projects/align_detr/), credits to [Zhi Cai](https://github.com/FelixCaae)
- Support [EVA-01 (CVPR'2023 Highlight)](https://github.com/baaivision/EVA/tree/master/EVA-01) and [EVA-02 (ArXiv'2023)](https://github.com/baaivision/EVA/tree/master/EVA-02) backbones, please check [DINO-EVA](./projects/dino_eva/) for more benchmarking results.

Please see [changelog.md](./changlog.md) for details and release history.

## Installation

Please refer to [Installation Instructions](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) for the details of installation.

## Getting Started

Please refer to [Getting Started with detrex](https://detrex.readthedocs.io/en/latest/tutorials/Getting_Started.html) for the basic usage of detrex. We also provides other tutorials for:
- [Learn about the config system of detrex](https://detrex.readthedocs.io/en/latest/tutorials/Config_System.html)
- [How to convert the pretrained weights from original detr repo into detrex format](https://detrex.readthedocs.io/en/latest/tutorials/Converters.html)
- [Visualize your training data and testing results on COCO dataset](https://detrex.readthedocs.io/en/latest/tutorials/Tools.html#visualization)
- [Analyze the model under detrex](https://detrex.readthedocs.io/en/latest/tutorials/Tools.html#model-analysis)
- [Download and initialize with the pretrained backbone weights](https://detrex.readthedocs.io/en/latest/tutorials/Using_Pretrained_Backbone.html)
- [Frequently asked questions](https://github.com/IDEA-Research/detrex/issues/109)
- [A simple onnx convert tutorial provided by powermano](https://github.com/IDEA-Research/detrex/issues/192)
- Simple training techniques: [Model-EMA](https://github.com/IDEA-Research/detrex/pull/201), [Mixed Precision Training](https://github.com/IDEA-Research/detrex/pull/198), [Activation Checkpoint](https://github.com/IDEA-Research/detrex/pull/200)
- [Simple tutorial about custom dataset training](https://github.com/IDEA-Research/detrex/pull/187)

Although some of the tutorials are currently presented with relatively simple content, we will constantly improve our documentation to help users achieve a better user experience.

## Documentation

Please see [documentation](https://detrex.readthedocs.io/en/latest/index.html) for full API documentation and tutorials.

## Model Zoo
Results and models are available in [model zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html).

<details open>
<summary> Supported methods </summary>

- [x] [DETR (ECCV'2020)](./projects/detr/)
- [x] [Deformable-DETR (ICLR'2021 Oral)](./projects/deformable_detr/)
- [x] [PnP-DETR (ICCV'2021)](./projects/pnp_detr/)
- [x] [Conditional-DETR (ICCV'2021)](./projects/conditional_detr/)
- [x] [Anchor-DETR (AAAI 2022)](./projects/anchor_detr/)
- [x] [DAB-DETR (ICLR'2022)](./projects/dab_detr/)
- [x] [DAB-Deformable-DETR (ICLR'2022)](./projects/dab_deformable_detr/)
- [x] [DN-DETR (CVPR'2022 Oral)](./projects/dn_detr/)
- [x] [DN-Deformable-DETR (CVPR'2022 Oral)](./projects/dn_deformable_detr/)
- [x] [Group-DETR (ArXiv'2022)](./projects/group_detr/)
- [x] [DETA (ArXiv'2022)](./projects/deta/)
- [x] [DINO (ICLR'2023)](./projects/dino/)
- [x] [H-Deformable-DETR (CVPR'2023)](./projects/h_deformable_detr/)
- [x] [MaskDINO (CVPR'2023)](./projects/maskdino/)
- [x] [CO-MOT (ArXiv'2023)](./projects/co_mot/)
- [x] [SQR-DETR (CVPR'2023)](./projects/sqr_detr/)
- [x] [Align-DETR (ArXiv'2023)](./projects/align_detr/)
- [x] [EVA-01 (CVPR'2023 Highlight)](./projects/dino_eva/)
- [x] [EVA-02 (ArXiv'2023)](./projects/dino_eva/)
- [x] [Focus-DETR (ICCV'2023)](./projects/focus_detr/)

Please see [projects](./projects/) for the details about projects that are built based on detrex.

</details>


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- detrex is an open-source toolbox for Transformer-based detection algorithms created by researchers of **IDEACVR**. We appreciate all contributions to detrex!
- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of its module design is borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr), and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).


## Citation
If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

- Citing **detrex**:

```BibTeX
@misc{ren2023detrex,
      title={detrex: Benchmarking Detection Transformers}, 
      author={Tianhe Ren and Shilong Liu and Feng Li and Hao Zhang and Ailing Zeng and Jie Yang and Xingyu Liao and Ding Jia and Hongyang Li and He Cao and Jianan Wang and Zhaoyang Zeng and Xianbiao Qi and Yuhui Yuan and Jianwei Yang and Lei Zhang},
      year={2023},
      eprint={2306.07265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<details>
<summary> Citing Supported Algorithms </summary>

```BibTex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}

@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}

@inproceedings{meng2021-CondDETR,
  title       = {Conditional DETR for Fast Training Convergence},
  author      = {Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year        = {2021}
}

@inproceedings{
      liu2022dabdetr,
      title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
      author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}

@inproceedings{li2022dn,
      title={Dn-detr: Accelerate detr training by introducing query denoising},
      author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13619--13627},
      year={2022}
}

@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{chen2022group,
  title={Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment},
  author={Chen, Qiang and Chen, Xiaokang and Wang, Jian and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2207.13085},
  year={2022}
}

@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}

@misc{li2022mask,
      title={Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation}, 
      author={Feng Li and Hao Zhang and Huaizhe xu and Shilong Liu and Lei Zhang and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2206.02777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{yan2023bridging,
 title={Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking},
 author={Yan, Feng and Luo, Weixin and Zhong, Yujie and Gan, Yiyang and Ma, Lin},
 journal={arXiv preprint arXiv:2305.12724},
 year={2023}
}

@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Fangyi and Zhang, Han and Hu, Kai and Huang, Yu-Kai and Zhu, Chenchen and Savvides, Marios},
    title     = {Enhanced Training of Query-Based Object Detection via Selective Query Recollection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23756-23765}
}
```


</details>



