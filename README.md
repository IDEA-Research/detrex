<h2 align="left">detrex</h2>
<p align="left">
    <a href="">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="">
        <img alt="GitHub" src="https://img.shields.io/github/license/Oneflow-Inc/libai.svg?color=blue">
    </a>
    <a href="">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="">
        <img alt="open issues" src="https://img.shields.io/github/issues-raw/Westlake-AI/openmixup?color=%23FF9600">
    </a>
    <a href="">
        <img alt="issue resolution" src="https://img.shields.io/badge/issue%20resolution-1%20d-%23009763">
    </a>
</p>

[📘Documentation]() |
[🛠️Installation]() |
[👀Model Zoo]() |
[🚀Awesome DETR](https://github.com/IDEA-Research/awesome-detection-transformer) |
[🆕News]() |
[🤔Reporting Issues](https://github.com/rentainhe/detrex/issues/new/choose)


## Introduction

detrex is an open-source toolbox that provides state-of-the-art transformer based detection algorithms on top of [Detectron2](https://github.com/facebookresearch/detectron2) and the module designs are partially borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection) and [DETR](https://github.com/facebookresearch/detr). Many thanks for their nicely organized code. The main branch works with **Pytorch 1.9+** or higher (we recommend **Pytorch 1.12**).


<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decompose the transformer based detection framework into various components which help the users to easily build their own customized models.

- **State-of-the-art methods.** detrex provides a series of transformer based detection algorithms including [DINO](https://arxiv.org/abs/2203.03605) which reach the new SOTA of DETR-like models with **63.3mAP**!

- **Easy to Use.** detrex is designed to be **light-weight** and easier for the users as follows:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight [training engine](./tools/train_net.py) modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)

Apart from detrex, we also released a repo [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer) to present papers about transformer for detection and segmentation.

</details>


## Installation

Please refer to [Installation Instructions]() for the details of installation.

## Getting Started

Please refer to [Getting Started with detrex]() for the basic usage of detrex.

## Documentation

Please see [documentation]() for full API documentation and tutorials.

## Model Zoo
Results and models are available in [model zoo]().

<details open>
<summary> Supported methods </summary>

- [x] [DETR (ECCV'2020)](./projects/detr/)
- [x] [Deformable-DETR (ICLR'2021)](./projects/dab_deformable_detr/)
- [x] [Conditional DETR (ICCV'2021)](./projects/conditional_detr/)
- [x] [DAB-DETR (ICLR'2022)](./projects/dab_detr/)
- [x] [DAB-Deformable-DETR (ICLR'2022)](./projects/dab_deformable_detr/)
- [x] [DN-DETR (CVPR'2022)](./projects/dn_detr/)
- [x] [DN-Deformable-DETR (CVPR'2022)](./projects/dn_deformable_detr/)
- [x] [DINO (ArXiv'2022)](./projects/dino/)

Please see [projects](./projects/) for the details about projects that are built based on detrex.

## Change Log

The beta v0.1.0 version was released in 30/09/2022. Highlights of the released version:
- Support various backbones including: [FocalNet](https://arxiv.org/abs/2203.11926), [Swin-T](https://arxiv.org/pdf/2103.14030.pdf), [ResNet](https://arxiv.org/abs/1512.03385) and other [detectron2 builtin backbones](https://github.com/facebookresearch/detectron2/tree/main/detectron2/modeling/backbone).
- Add [timm](https://github.com/rwightman/pytorch-image-models) backbones wrapper and [torchvision](https://github.com/pytorch/vision) backbones wrapper.
- Support various transformer based detection algorithms including: [DETR](https://arxiv.org/abs/2005.12872), [Deformable-DETR](https://arxiv.org/abs/2010.04159), [Conditional-DETR](https://arxiv.org/abs/2108.06152), [DAB-DETR](https://arxiv.org/abs/2201.12329), [DN-DETR](https://arxiv.org/abs/2203.01305), [DINO](https://arxiv.org/abs/2203.03605).
- Support flexible config system based on [Lazy Configs](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)

Please see [changelog.md](./changlog.md) for details and release history.

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- detrex is an open-source toolbox for transformer based detection algorithms created by researchers of **IDEACVR**. We appreciate all contributions to detrex!
- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of the module design borrows from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr) and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).


## Citation
