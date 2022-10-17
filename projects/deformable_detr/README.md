## Deformable DETR: Deformable Transformers for End-to-End Object Detection

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai

[[`arXiv`](https://arxiv.org/abs/2010.04159)] [[`BibTeX`](#citing-deformable-detr)]


<div align="center">
  <img src="./assets/deformable_detr.png"/>
</div><br/>

## Pretrained Weights
Here we provide the pretrained `Deformable-DETR` weights based on detrex.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: deformable_detr_r50_50ep -->
 <tr><td align="left"> <a href="configs/deformable_detr_r50_50ep.py"> Deformable-DETR </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center"></td>
<td align="center"> <a href=""> model </a></td>
</tr>
<!-- ROW: deformable_detr_r50_with_box_refinement_50ep -->
 <tr><td align="left"><a href="configs/deformable_detr_r50_with_box_refinement_50ep.py">Deformable-DETR-R50 + Box-Refinement</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.99</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_with_box_refinement_50ep_new.pth">model</a></td>
</tr>
<!-- ROW: deformable_detr_r50_two_stage_50ep -->
 <tr><td align="left"><a href="configs/deformable_detr_r50_two_stage_50ep.py">Deformable-DETR-R50 + Box-Refinement + Two-Stage</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">48.19</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_r50_two_stage_50ep_new.pth">model</a></td>
</tr>
</tbody></table>

All the models are trained using `8 GPUs` with total batch size equals to `16`. We've observed that the result of `deformable-two-stage` model trained using `8 GPUs` may be  slightly lower than `16 GPUs` with `32` total batch size.

**Notable facts and caveats**: The training settings are different from the original repo, we use `lr=1e-5` for backbone and `1e-4` for the other modules. The original implementation sets `lr` to `2e-5` for `backbone`, `sampling_offsets` and `reference_points`, and `2e-4` for other modules. And we used `top-300` confidence boxes for testing, which may get a slightly better results on COCO evaluation. And we only freeze the stem layer in ResNet backbone by setting `freeze_at=1` in config.

## Converted Weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: deformable_detr_r50_50ep -->
 <tr><td align="left"><a href="configs/deformable_detr_r50_50ep.py">Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">44.59</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_deformable_detr_r50.pth">model</a></td>
</tr>
<!-- ROW: deformable_detr_r50_with_box_refinement -->
 <tr><td align="left"><a href="configs/deformable_detr_r50_with_box_refinement_50ep.py">Deformable-DETR-R50 + Box-Refinement</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.28</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_deformable_detr_r50_with_box_refine_50ep.pth">model</a></td>
</tr>
<!-- ROW: deformable_detr_r50_two_stage_50ep -->
 <tr><td align="left"><a href="configs/deformable_detr_r50_two_stage_50ep.py">Deformable-DETR-R50 + Box-Refinement + Two-Stage</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.09</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_deformable_detr_two_stage_50ep.pth">model</a></td>
</tr>
</tbody></table>

**Note:** Here we borrowed the pretrained weight from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) official repo. And all the pretrained weights are tested using `top-300` confidence boxes (`top-100` in original repo) which may brings about `0.2 AP` gain on COCO evaluation.

## Training
All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/deformable_detr/configs/path/to/config.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 16 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/deformable_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```

## Citing Deformable-DETR
```BibTex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```