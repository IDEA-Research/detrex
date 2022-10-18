## Change Log

### v0.1.1 (18/10/2022)
#### New Features
- Add model analyze tools for detrex [#79](https://github.com/IDEA-Research/detrex/pull/79)
- Add benchmark [#81](https://github.com/IDEA-Research/detrex/pull/81)
- Add visualization for COCO eval results and annotations [#82](https://github.com/IDEA-Research/detrex/pull/82)
- Support `Group-DETR` algorhtim [#84](https://github.com/IDEA-Research/detrex/pull/84)
- Release `DINO-Swin` training results [#86](https://github.com/IDEA-Research/detrex/pull/86)
- Release better `Deformable-DETR` baselines [#102](https://github.com/IDEA-Research/detrex/pull/102) [#103](https://github.com/IDEA-Research/detrex/pull/103) 

#### Bug Fixes
- Fix bugs in ConvNeXt backbone [#91](https://github.com/IDEA-Research/detrex/pull/91)

#### Documentation
- Add pretrained model weights download links [#86](https://github.com/IDEA-Research/detrex/pull/86)

### v0.1.0 (21/09/2022)
The **beta v0.1.0** version of detrex was released in 21/09/2022

#### New Features
- Support various backbones including: [FocalNet](https://arxiv.org/abs/2203.11926), [Swin-T](https://arxiv.org/pdf/2103.14030.pdf), [ResNet](https://arxiv.org/abs/1512.03385) and other [detectron2 builtin backbones](https://github.com/facebookresearch/detectron2/tree/main/detectron2/modeling/backbone).
- Add [timm](https://github.com/rwightman/pytorch-image-models) backbones wrapper and [torchvision](https://github.com/pytorch/vision) backbones wrapper.
- Support various transformer based detection algorithms including: [DETR](https://arxiv.org/abs/2005.12872), [Deformable-DETR](https://arxiv.org/abs/2010.04159), [Conditional-DETR](https://arxiv.org/abs/2108.06152), [DAB-DETR](https://arxiv.org/abs/2201.12329), [DN-DETR](https://arxiv.org/abs/2203.01305), [DINO](https://arxiv.org/abs/2203.03605).
- Support flexible config system based on [Lazy Configs](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)