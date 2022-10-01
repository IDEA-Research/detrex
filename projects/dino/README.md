## DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum

[[`arXiv`](https://arxiv.org/abs/2203.03605)] [[`BibTeX`](#citing-dino)]

<div align="center">
  <img src="./assets/dino_arch.png"/>
</div><br/>

## Pretrained Models
Here we provide the pretrained `DINO` weights based on detrex.
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
<!-- ROW: dino_r50_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino_r50_4cale_12ep.py">DINO-R50-4scale</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">49.05</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dino_r50_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_r101_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino_r101_4cale_12ep.py">DINO-R101-4scale</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center"></td>
<td align="center"> <a href="">model</a></td>
</tr>
<!-- ROW: dino_swin_tiny_4cale_12ep -->
 <tr><td align="left"><a href="configs/dino_swin_tiny_4cale_12ep.py">DINO-Swin-T-224-4scale</a></td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center"></td>
<td align="center"> <a href="">model</a></td>
</tr>
<!-- ROW: dino_swin_tiny_4cale_12ep -->
 <tr><td align="left"><a href="configs/dino_swin_tiny_4cale_12ep.py">DINO-Swin-T-224-4scale</a></td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center"></td>
<td align="center"> <a href="">model</a></td>
</tr>
<!-- ROW: dino_swin_base_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino_swin_base_384_4scale_12ep.py">DINO-Swin-B-384-4scale</a></td>
<td align="center">Swin-Base-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">55.83</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth">model</a></td>
</tr>
<!-- ROW: dino_swin_large_4scale_12ep -->
 <tr><td align="left"><a href="configs/dino_swin_large_384_4scale_12ep.py">DINO-Swin-L-384-4scale</a></td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">56.93</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_4scale_12ep.pth">model</a></td>
</tr>
</tbody></table>

**Note**: `Swin-X-384` means the pretrained resolution is `384 x 384` and `IN22k to In1k` means the model is pretrained on `ImageNet-22k` and finetuned on `ImageNet-1k`.

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