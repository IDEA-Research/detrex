## Change Log

### v0.3.0 (17/03/2023)
- Support new algorithms including `Anchor-DETR` and `DETA`.
- Release more than 10+ pretrained models (including the converted weights): `DETR-R50 & R101`, `DETR-R50 & R101-DC5`, `DAB-DETR-R50 & R101-DC5`, `DAB-DETR-R50-3patterns`, `Conditional-DETR-R50 & R101-DC5`, `DN-DETR-R50-DC5`, `Anchor-DETR` and the `DETA-Swin-o365-finetune` model which can achieve **`62.9AP`** on coco val.
- Support **MaskDINO** on ADE20k semantic segmentation task.
- Support `EMAHook` during training by setting `train.model_ema.enabled=True`, which can enhance the model performance. DINO with EMA can achieve **`49.4AP`** with only 12epoch training.
- Support mixed precision training by setting `train.amp.enabled=True`, which will **reduce 20% to 30% GPU memory usage**.
- Support `train.fast_dev_run=True` for **fast debugging**.
- Support **encoder-decoder checkpoint** in DINO, which may reduce **30% GPU** memory usage.
- Support a great slurm training scripts by @rayleizhu, please check this issue for more details [#213](https://github.com/IDEA-Research/detrex/issues/213)


### v0.2.1 (01/02/2023)
#### New Algorithm
- MaskDINO COCO instance-seg/panoptic-seg pre-release [#154](https://github.com/IDEA-Research/detrex/pull/154)

#### New Features
- New baselines for `Res/Swin-DINO-5scale`, `ViTDet-DINO`, `FocalNet-DINO`, etc. [#138](https://github.com/IDEA-Research/detrex/pull/138), [#155](https://github.com/IDEA-Research/detrex/pull/155)
- Support FocalNet backbone [#145](https://github.com/IDEA-Research/detrex/pull/145)
- Support Swin-V2 backbone [#152](https://github.com/IDEA-Research/detrex/pull/152)

#### Documentation
- Add ViTDet / FocalNet download links and usage example, please refer to [Download Pretrained Weights](https://detrex.readthedocs.io/en/latest/tutorials/Download_Pretrained_Weights.html).
- Add tutorial on how to verify the correct installation of detrex. [#194](https://github.com/IDEA-Research/detrex/pull/194)

#### Bug Fixes
- Fix demo confidence filter not to remove mask predictions [#156](https://github.com/IDEA-Research/detrex/pull/156)

#### Code Refinement
- Make more readable logging info for criterion and matcher [#151](https://github.com/IDEA-Research/detrex/pull/151)
- Modified learning rate scheduler config usage, add fundamental scheduler configuration [#191](https://github.com/IDEA-Research/detrex/pull/191)

### v0.2.0 (13/11/2022)
#### New Features
- Rebuild cleaner config files for projects [#107](https://github.com/IDEA-Research/detrex/pull/107)
- Support [H-Deformable-DETR](https://github.com/IDEA-Research/detrex/tree/main/projects/h_deformable_detr) [#110](https://github.com/IDEA-Research/detrex/pull/110)
- Release H-Deformable-DETR pretrained weights including `H-Deformable-DETR-R50`, `H-Deformable-DETR-Swin-Tiny`, `H-Deformable-DETR-Swin-Large`.
- Add demo for visualizing customized input images or videos using pretrained weights [#119](https://github.com/IDEA-Research/detrex/pull/119)
- Release new baselines for `DINO-Swin-Large-36ep`, `DAB-Deformable-DETR-R50-50ep`, `DAB-Deformable-DETR-Two-Stage-50ep`, `H-DETR`.

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

### v0.1.0 (30/09/2022)
The **beta v0.1.0** version of detrex was released in 30/09/2022

#### New Features
- Support various backbones including: [FocalNet](https://arxiv.org/abs/2203.11926), [Swin-T](https://arxiv.org/pdf/2103.14030.pdf), [ResNet](https://arxiv.org/abs/1512.03385) and other [detectron2 builtin backbones](https://github.com/facebookresearch/detectron2/tree/main/detectron2/modeling/backbone).
- Add [timm](https://github.com/rwightman/pytorch-image-models) backbones wrapper and [torchvision](https://github.com/pytorch/vision) backbones wrapper.
- Support various transformer based detection algorithms including: [DETR](https://arxiv.org/abs/2005.12872), [Deformable-DETR](https://arxiv.org/abs/2010.04159), [Conditional-DETR](https://arxiv.org/abs/2108.06152), [DAB-DETR](https://arxiv.org/abs/2201.12329), [DN-DETR](https://arxiv.org/abs/2203.01305), [DINO](https://arxiv.org/abs/2203.03605).
- Support flexible config system based on [Lazy Configs](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)