# H-Deformable-DETR

This is the official implementation of the paper "[DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)". 

Authors: Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, Han Hu

## Pretrained Models
Here we provide the H-Deformable-DETR model pretrained weights based on detrex:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Query Num</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_r50_two_stage_12ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">49.1</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/h_deformable_detr_r50_two_stage_12ep_modified_train_net.pth">model</a></td>
</tr>
</tbody></table>

We prefer the users to use the modified [train_net.py](./train_net.py) to reproduce the results:
```python
python projects/h_deformable_detr/train_net.py --config-file projects/h_deformable_detr/configs/path/to/config.py --num-gpus 8
```

## Converted Models

We provide a set of baseline results and trained models available for download:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Query Num</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_r50_two_stage_12ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">48.9</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_r50_two_stage_36ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">50.3</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
<tr><td align="left"><a href="configs/h_deformable_detr_swin_tiny_two_stage_12ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">50.6</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_swin_tiny_two_stage_36ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">53.5</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_swin_large_two_stage_12ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">56.2</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_swin_large_two_stage_36ep.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">57.5</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_swin_large_two_stage_12ep_900queries.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">900</td>
<td align="center">12</td>
<td align="center">56.4</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/h_deformable_detr_swin_large_two_stage_36ep_900queries.py">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">900</td>
<td align="center">36</td>
<td align="center">57.7</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tbody></table>



## Run
### Training

All configs can be trained with:

```bash
cd detrex
python tools/train_net.py --config-file projects/h_deformable_detr/configs/path/to/config.py --num-gpus 8
```

* By default, we use 8 GPUs with total batch size as 16 for training.
* To train/eval a model with the swin transformer backbone, you need to download the backbone from the [offical repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models) frist and specify argument `train.init_checkpoint` like [our configs](./configs/h_deformable_detr_swin_tiny_two_stage_12ep.py).
* It's better to set `loss_class_weight=1.0` when using the default training engine.

### Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/h_deformable_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```




## Citing H-Deformable-DETR
If you find H-Deformable-DETR useful in your research, please consider citing:

```bibtex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}

@article{zhu2020deformable,
  title={Deformable detr: Deformable transformers for end-to-end object detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```