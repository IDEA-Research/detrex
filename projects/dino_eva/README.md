## DINO + EVA Series

We implement [DINO](https://arxiv.org/abs/2203.03605) with [EVA](https://github.com/baaivision/EVA) backbone and [LSJ augmentation](https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet/configs/common/coco_loader_lsj.py) in this project, the original DINO project can be found in [DINO Project](../dino/)

[[`DINO ArXiv`](https://arxiv.org/abs/2203.03605)] [[`EVA ArXiv`](https://arxiv.org/abs/2211.07636)] [[`BibTeX`](#citing-dino-and-eva)]


## Table of Contents
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citing-dino-and-eva)

## Pretrained Models

<div align="center">

| Name | init. model weight | LSJ crop size | epoch | AP box | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dino-eva-02` | `eva02_L_m38m_to_o365` | `1536x1536` | 12 | 61.6 | [config](./configs/dino-eva-02/dino_eva_02_vitdet_l_8attn_1536_lrd0p8_4scale_12ep.py) | [Huggingface](https://huggingface.co/IDEA-CVR/detrex/resolve/main/dino_eva_02_o365_backbone_finetune_vitdet_l_8attn_lsj_1536_4scale_12ep.pth) |
| `dino-eva-02` | `eva02_L_pt_m38m_p14to16` | `1024x1024` | 12 | 58.9 | [config](./configs/dino-eva-02/dino_eva_02_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.py) | [Huggingface](https://huggingface.co/IDEA-CVR/detrex/resolve/main/dino_eva_02_m38m_pretrain_vitdet_l_4attn_1024_lrd0p8_4scale_12ep.pth) |

</div>

- For `o365` pretrained EVA model we only load its backbone weights.
- All the pretrained EVA weights can be downloaded from [here](https://github.com/baaivision/EVA).
- All `1x settings (12 epochs)` models were trained using the hacked [train_net.py](./train_net.py)

## Training
For `1x settings (12 epochs)`, we trained them using the hacked [train_net.py](./train_net.py), here's the training scripts:
```bash
cd detrex
python projects/dino_eva/train_net.py --config-file projects/dino_eva/configs/path/to/config.py --num-gpus 8
```

All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/dino_eva/configs/path/to/config.py --num-gpus 8
```
The configs can be chosen from [dino-eva-01](./configs/dino-eva-01/) or [dino-eva-02](./configs/dino-eva-02/). By default, we use 8 GPUs with total batch size as 16 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/dino_eva/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


## Citing DINO and EVA
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

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```
