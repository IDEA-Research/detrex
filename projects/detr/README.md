## DETR: End-to-End Object Detection with Transformers

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko

<div align="center">
  <img src="./assets/DETR.png"/>
</div><br/>

We reproduce DETR in detrex based on [Detectron2 wrapper for DETR](https://github.com/facebookresearch/detr/tree/main/d2).

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