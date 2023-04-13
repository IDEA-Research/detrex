## DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum

[[`arXiv`](https://arxiv.org/abs/2203.03605)] [[`BibTeX`](#citing-dino)]

<div align="center">
  <img src="./assets/dino_arch.png"/>
</div><br/>

## DINO with modified training engine
We've provide a hacked [train_net.py](./train_net.py) which aligns the optimizer params with Deformable-DETR that can achieve a better result on DINO models.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: dino_r50_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino_r50_4scale_12ep.py">DINO-R50-4scale (hacked trainer)</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.4</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_r50_4scale_12ep_hacked_trainer.pth">model</a></td>
</tr>
</tbody></table>

- Training model with hacked trainer
```python
python projects/dino/train_net.py --config-file /path/to/config.py --num-gpus 8
```

## Main Results with Pretrained Models

**Pretrained DINO with ResNet Backbone**

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: dino_r50_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-resnet/dino_r50_4scale_12ep.py">DINO-R50-4scale</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_r50_4scale_12ep_49_2AP.pth">model</a></td>
</tr>
 <tr><td align="left">DINO-R50-4scale <b> with AMP</b></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.1</td>
<td align="center"> - </td>
</tr>
 <tr><td align="left">DINO-R50-4scale <b> with EMA</b></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.4</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/dino_r50_4scale_12ep_with_ema.pth">model</a> </td>
</tr>
<!-- ROW: dino_r50_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-resnet/dino_r50_5scale_12ep.py">DINO-R50-5scale</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.6</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_r50_5scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_r50_4scale_12ep_300dn -->
 <tr><td align="left"><a href="configs/dino-resnet/dino_r50_4scale_12ep_300dn.py">DINO-R50-4scale</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">300</td>
<td align="center">49.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_r50_4scale_12ep_300dn.pth">model</a></td>
</tr>
<!-- ROW: dino_r50_4scale_24ep -->
 <tr><td align="left"><a href="configs/dino-resnet/dino_r50_4scale_24ep.py">DINO-R50-4scale</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">24</td>
<td align="center">100</td>
<td align="center">50.6</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r50_4scale_24ep.pth">model</a></td>
</tr>
<!-- ROW: dino_r101_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-resnet/dino_r101_4scale_12ep.py">DINO-R101-4scale</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">50.0</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r101_4scale_12ep.pth">model</a></td>
</tr>
</tbody></table>

**Pretrained DINO with Swin-Transformer Backbone**
<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- ROW: dino_swin_tiny_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_tiny_4scale_12ep.py">DINO-Swin-T-224-4scale</a></td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">51.3</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_tiny_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_tiny_4scale_12ep.py">DINO-Swin-T-224-4scale</a></td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_small_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_small_224_4scale_12ep.py">DINO-Swin-S-224-4scale</a></td>
<td align="center">Swin-Small-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">53.0</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_small_224_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_base_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_base_384_4scale_12ep.py">DINO-Swin-B-384-4scale</a></td>
<td align="center">Swin-Base-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">55.8</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_large_224_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_large_224_4scale_12ep.py">DINO-Swin-L-224-4scale</a></td>
<td align="center">Swin-Large-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">56.9</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_224_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_large_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_large_384_4scale_12ep.py">DINO-Swin-L-384-4scale</a></td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">56.9</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_large_384_5scale_12ep.py">DINO-Swin-L-384-5scale</a></td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">57.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_swin_large_384_5scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_large_4scale_36ep -->
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_large_384_4scale_36
 ep.py">DINO-Swin-L-384-4scale</a></td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">36</td>
<td align="center">100</td>
<td align="center">58.1</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_swin_large_384_4scale_36ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-swin/dino_swin_large_384_5scale_36
 ep.py">DINO-Swin-L-384-5scale</a></td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">36</td>
<td align="center">100</td>
<td align="center">58.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_swin_large_384_5scale_36ep.pth">model</a></td>
</tr>
</tbody></table>

**Pretrained DINO with FocalNet Backbone**
<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
 <tr><td align="left"><a href="configs/dino-focal/dino_focalnet_large_lrf_384_4scale_12ep
 ep.py">DINO-Focal-Large-4scale</a></td>
<td align="center">FocalNet-384-LRF-3Level</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">57.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focal_large_lrf_384_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-focal/dino_focalnet_large_lrf_384_4scale_36ep
 ep.py">DINO-Focal-Large-4scale</a></td>
<td align="center">FocalNet-384-LRF-3Level</td>
<td align="center">IN22k</td>
<td align="center">36</td>
<td align="center">100</td>
<td align="center">58.3</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_focal_large_3level_4scale_36ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-focal/dino_focalnet_large_lrf_384_4scale_12ep
 ep.py">DINO-Focal-Large-4scale</a></td>
<td align="center">FocalNet-384-LRF-4Level</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">58.0</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focal_large_lrf_384_fl4_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-focal/dino_focalnet_large_lrf_384_5scale_12ep
 ep.py">DINO-Focal-Large-5scale</a></td>
<td align="center">FocalNet-384-LRF-4Level</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">58.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focalnet_large_lrf_384_fl4_5scale_12ep.pth">model</a></td>
</tr>
</tbody></table>

**Pretrained DINO with ViT Backbone**
<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
 <tr><td align="left"><a href="configs/dino-vitdet/dino_vitdet_base_4scale_12ep
 ep.py">DINO-ViTDet-Base-4scale</a></td>
<td align="center">ViT</td>
<td align="center">IN1k, MAE</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">50.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-vitdet/dino_vitdet_base_4scale_50ep
 ep.py">DINO-ViTDet-Base-4scale</a></td>
<td align="center">ViT</td>
<td align="center">IN1k, MAE</td>
<td align="center">50</td>
<td align="center">100</td>
<td align="center">55.0</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_base_4scale_50ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-vitdet/dino_vitdet_large_4scale_12ep
 ep.py">DINO-ViTDet-Large-4scale</a></td>
<td align="center">ViT</td>
<td align="center">IN1k, MAE</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.9</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_large_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-vitdet/dino_vitdet_large_4scale_50ep
 ep.py">DINO-ViTDet-Large-4scale</a></td>
<td align="center">ViT</td>
<td align="center">IN1k, MAE</td>
<td align="center">50</td>
<td align="center">100</td>
<td align="center">57.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_large_4scale_50ep.pth">model</a></td>
</tr>
</tbody></table>

**Pretrained DINO with ConvNeXt Backbone**
<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
 <tr><td align="left"><a href="configs/dino-internimage/dino_convnext_tiny_384_4scale_12ep.py">DINO-ConvNeXt-Tiny-384-4scale</a></td>
<td align="center">ConvNeXt-Tiny-384</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.4</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_convnext_tiny_384_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-internimage/dino_convnext_small_384_4scale_12ep.py">DINO-ConvNeXt-Small-384-4scale</a></td>
<td align="center">ConvNeXt-Small-384</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">54.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_convnext_small_384_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-internimage/dino_convnext_base_384_4scale_12ep.py">DINO-ConvNeXt-Base-384-4scale</a></td>
<td align="center">ConvNeXt-Base-384</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">55.1</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_convnext_base_384_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-convnext/dino_convnext_large_384_4scale_12ep.py">DINO-ConvNeXt-Large-384-4scale</a></td>
<td align="center">ConvNeXt-Large-384</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">55.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_convnext_large_384_4scale_12ep.pth">model</a></td>
</tr>
</tbody></table>


**Pretrained DINO with InternImage Backbone**
<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
 <tr><td align="left"><a href="configs/dino-internimage/dino_internimage_tiny_4scale_12ep.py">DINO-InternImage-Tiny-4scale</a></td>
<td align="center">InternImage-Tiny</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.3</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_internimage_tiny_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-internimage/dino_internimage_small_4scale_12ep.py">DINO-InternImage-Small-4scale</a></td>
<td align="center">InternImage-Small</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">53.6</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_internimage_small_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-internimage/dino_internimage_base_4scale_12ep.py">DINO-InternImage-Base-4scale</a></td>
<td align="center">InternImage-Base</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">54.7</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_internimage_base_4scale_12ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/dino-internimage/dino_internimage_large_4scale_12ep.py">DINO-InternImage-Large-4scale</a></td>
<td align="center">InternImage-Large</td>
<td align="center">IN22k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">57.0</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/dino_internimage_large_4scale_12ep.pth">model</a></td>
</tr>
</tbody></table>



**Note**: 
- `Swin-X-384` means the backbone pretrained resolution is `384 x 384` and `IN22k to In1k` means the model is pretrained on `ImageNet-22k` and finetuned on `ImageNet-1k`.
- ViT backbone using MAE pretraining weights following [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)  which can be downloaded in [MAE](https://github.com/facebookresearch/mae). And it's not stable to train ViTDet-DINO without warmup lr-scheduler.
- `Focal-LRF-3Level`: means using `Large-Receptive-Field (LRF)` and `Focal-Level` is setted to `3`, please refer to [FocalNet](https://github.com/microsoft/FocalNet) for more details about the backbone settings.
- `with AMP`: means using mixed precision training.
- `with EMA`: means training with model **E**xponential **M**oving **A**verage (EMA).

**Notable facts and caveats**: The position embedding of DINO in detrex is different from the original repo. We set the tempureture and offsets in `PositionEmbeddingSine` to `10000` and `-0.5` which may make the model converge a little bit faster in the early stage and get a slightly better results (about 0.1mAP) in 12 epochs settings.


## Training
All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/dino/configs/path/to/config.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 16 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/dino/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


## Citing DINO
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTex
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
