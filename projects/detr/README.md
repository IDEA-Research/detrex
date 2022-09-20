## DETR: End-to-End Object Detection with Transformers

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko

[[`arXiv`](https://arxiv.org/abs/2005.12872)] [[`BibTeX`](#citing-detr)]

<div align="center">
  <img src="./assets/DETR.png"/>
</div><br/>

We reproduce DETR in detrex based on [Detectron2 wrapper for DETR](https://github.com/facebookresearch/detr/tree/main/d2).

## Converted Weights
Here we provides the weights which are converted by [converter.py](./converter.py) from the official DETR repo.

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
<!-- ROW: detr_r50 -->
 <tr><td align="left"><a>DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">500</td>
<td align="center">42.01</td>
<td align="center"> <a href="">model</a></td>
</tr>
<!-- ROW: detr_r50 -->
 <tr><td align="left"><a>DETR-R101</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">500</td>
<td align="center">42.01</td>
<td align="center"> <a href="">model</a></td>
</tr>
</tbody></table>

## Training
Training DETR model for 300 epochs:
```bash
cd detrex
python tools/train_net.py --config-file projects/detr/configs/detr_r50_300ep.py --num-gpus 8
```
By default, we use 8 GPUs with total batch size as 64 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/detr/configs/path/to/config.py \
    --eval-only train.init_checkpoint=/path/to/model_checkpoint
```

## Evaluating the official DETR model
Using the modified conversion script to convert models trained by the official [DETR](https://github.com/facebookresearch/detr) training loop into the format of detrex model. To download and evaluate `DETR-R50` model, simply run:
```bash
cd detrex
python projects/detr/converter.py \
    --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_model converted_detr_r50_model.pth
```
Then evaluate the converted model like:
```bash
python tools/train_net.py --config-file projects/detr/configs/detr_r50_300ep.py \
    --eval-only train.init_checkpoint="./converted_detr_r50_model.pth"
```

## Citing DETR
```BibTex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```