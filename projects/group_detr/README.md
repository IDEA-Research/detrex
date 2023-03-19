## Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment

Chen, Qiang and Chen, Xiaokang and Wang, Jian and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong

[[`arXiv`](https://arxiv.org/abs/2207.13085)] [[`BibTeX`](#citing-group-detr)]

<div align="center">
  <img src="./assets/group_detr_arch.png"/>
</div><br/>

**Note**: This is the implementation of `Conditional DETR + Group DETR`

## Converted Models
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
<!-- ROW: dn_detr_r50_50ep -->
 <tr><td align="left"><a href="configs/dn_detr_r50_dc5_50ep.py">Group-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">37.8</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/converted_group_detr_r50_12ep.pth">model</a></td>
</tr>
</tr>
</tbody></table>

## Training
All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/group_detr/configs/path/to/config.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 16 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/group_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```

## Citing Group-DETR
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTex
@article{chen2022group,
  title={Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment},
  author={Chen, Qiang and Chen, Xiaokang and Wang, Jian and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2207.13085},
  year={2022}
}
```
