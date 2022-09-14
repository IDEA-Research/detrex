<h2 align="center">detrex</h2>
<p align="center">
    <a href="">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="">
        <img alt="GitHub" src="https://img.shields.io/github/license/Oneflow-Inc/libai.svg?color=blue">
    </a>
    <a href="https://github.com/Oneflow-Inc/libai/issues">
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
[🚀Awesome DETR](https://github.com/IDEACVR/awesome-detection-transformer) |
[🆕News]() |
[🤔Reporting Issues](https://github.com/rentainhe/detrex/issues/new/choose)


## Introduction

`detrex` is an open-source toolbox that provides state-of-the-art transformer based detection algorithms on top of [Detectron2](https://github.com/facebookresearch/detectron2) and the module designs are partially borrowed from [mmdetection](https://github.com/open-mmlab/mmdetection) and [detr official repo](https://github.com/facebookresearch/detr). Many thanks for the nicely organized code. The main branch works with **Pytorch 1.9+** or higher (we recommend **Pytorch 1.12**).


<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decompose the transformer based detection framework into various components which help the users to easily build their own customized models.

- **State-of-the-art methods.** detrex provides a series of transformer based detection algorithms including `DINO` which reach the new SOTA of DETR-like models with **63.3mAP**!

- **Easy to Use.** detrex is designed to be **light-weight** and easier for the users as follows:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight training engine modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)



</details>