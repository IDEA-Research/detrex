## Anchor DETR: Query Design for Transformer-Based Object Detection

Yingming Wang, Xiangyu Zhang, Tong Yang, Jian Sun

[[`arXiv`](https://arxiv.org/abs/2109.07107)] [[`BibTeX`](#citing-anchor-detr)]

<div align="center">
  <img src="./assets/anchor_detr_arch.png"/>
</div><br/>

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
 <tr><td align="left"><a href="configs/anchor_detr_r50_50ep.py">Anchor-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">42.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r50_50ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/detr_r50_dc5_300ep.py">Anchor-R50-DC5</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">44.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r50_dc5_50ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/detr_r101_300ep.py">Anchor-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">43.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r101_50ep.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/detr_r101_dc5_300ep.py">DETR-R101-DC5</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.1</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r101_dc5_50ep.pth">model</a></td>
</tr>
</tbody></table>

**Note:** Here we borrowed the pretrained weight from [Anchor-DETR](https://github.com/megvii-research/AnchorDETR) official repo. And our detrex training results will be released in the future version.

## Training
Training Anchor-DETR-R50 model:
```bash
cd detrex
python tools/train_net.py --config-file projects/anchor_detr/configs/anchor_detr_r50_50ep.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 64 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/anchor_detr/configs/path/to/config.py \
    --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


## Citing Anchor-DETR
```BibTex
@inproceedings{wang2022anchor,
  title={Anchor detr: Query design for transformer-based detector},
  author={Wang, Yingming and Zhang, Xiangyu and Yang, Tong and Sun, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={2567--2575},
  year={2022}
}
```