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
[🆕News](#change-log) |
[🤔Reporting Issues](https://github.com/IDEA-Research/detrex/issues/new/choose)


## Introduction

detrex is an open-source toolbox that provides state-of-the-art Transformer-based detection algorithms. It is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and its module design is partially borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection) and [DETR](https://github.com/facebookresearch/detr). Many thanks for their nicely organized code. The main branch works with **Pytorch 1.10+** or higher (we recommend **Pytorch 1.12**).

<div align="center">
  <img src="./assets/detr_arch.png" width="100%"/>
</div>

<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decomposes the Transformer-based detection framework into various components which help users easily build their own customized models.

- **State-of-the-art Methods.** detrex provides a series of Transformer-based detection algorithms, including [DINO](https://arxiv.org/abs/2203.03605) which reached the SOTA of DETR-like models with **63.3mAP**!

- **Easy to Use.** detrex is designed to be **light-weight** and easy for users to use:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight [training engine](./tools/train_net.py) modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)

Apart from detrex, we also released a repo [Awesome Detection Transformer](https://github.com/IDEA-Research/awesome-detection-transformer) to present papers about Transformer for detection and segmentation.

</details>

## Fun Facts
The repo name detrex has several interpretations:
- <font color=blue> <b> detr-ex </b> </font>: We take our hats off to DETR and regard this repo as an extension of Transformer-based detection algorithms.

- <font color=#db7093> <b> det-rex </b> </font>: rex literally means 'king' in Latin. We hope this repo can help advance the state of the art on object detection by providing the best Transformer-based detection algorithms from the research community.

- <font color=#008000> <b> de-t.rex </b> </font>: de means 'the' in Gemany. T.rex means 'king of the tyrant lizards' and connects to our research work 'DINO', which is short for Dinosaur.


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
- [x] [Deformable-DETR (ICLR'2021 Oral)](./projects/dab_deformable_detr/)
- [x] [Conditional DETR (ICCV'2021)](./projects/conditional_detr/)
- [x] [DAB-DETR (ICLR'2022)](./projects/dab_detr/)
- [x] [DAB-Deformable-DETR (ICLR'2022)](./projects/dab_deformable_detr/)
- [x] [DN-DETR (CVPR'2022 Oral)](./projects/dn_detr/)
- [x] [DN-Deformable-DETR (CVPR'2022 Oral)](./projects/dn_deformable_detr/)
- [x] [DINO (ArXiv'2022)](./projects/dino/)

Please see [projects](./projects/) for the details about projects that are built based on detrex.

</details>


## Change Log

The **beta v0.1.0** version was released in 30/09/2022. Highlights of the released version:
- Support various backbones, including: [FocalNet](https://arxiv.org/abs/2203.11926), [Swin-T](https://arxiv.org/pdf/2103.14030.pdf), [ResNet](https://arxiv.org/abs/1512.03385) and other [detectron2 builtin backbones](https://github.com/facebookresearch/detectron2/tree/main/detectron2/modeling/backbone).
- Add [timm](https://github.com/rwightman/pytorch-image-models) backbone wrapper and [torchvision](https://github.com/pytorch/vision) backbone wrapper.
- Support various Transformer-based detection algorithms, including: [DETR](https://arxiv.org/abs/2005.12872), [Deformable-DETR](https://arxiv.org/abs/2010.04159), [Conditional-DETR](https://arxiv.org/abs/2108.06152), [DAB-DETR](https://arxiv.org/abs/2201.12329), [DN-DETR](https://arxiv.org/abs/2203.01305), and [DINO](https://arxiv.org/abs/2203.03605).
- Support flexible config system based on [Lazy Configs](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)

Please see [changelog.md](./changlog.md) for details and release history.

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- detrex is an open-source toolbox for Transformer-based detection algorithms created by researchers of **IDEACVR**. We appreciate all contributions to detrex!
- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of its module design is borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr), and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).


## Citation
If you find this project useful in your research, please consider cite:
```BibTeX
@misc{ren2022detrex,
  author =       {Tianhe Ren and Shilong Liu and Hao Zhang and
                  Feng Li and Xingyu Liao and Lei Zhang},
  title =        {detrex},
  howpublished = {\url{https://github.com/IDEA-Research/detrex}},
  year =         {2022}
}
```