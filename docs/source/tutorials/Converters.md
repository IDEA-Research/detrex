# Convert Pretrained Models
This document provides a brief intro of how to convert the pretrained model into the format of detrex.


## Convert TorchVision Pretrained ResNet Models
To use the detectron2 provided pretrained weights, please refer to [ImageNet Pretrained Models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#imagenet-pretrained-models). Here we've noticed that detectron2 only provided a converted torchvision `ResNet-50` model. For more pretrained models like `ResNet{101, 152}`. You can use the detectron2 provided [conversion script](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py) to convert the torchvision pretrained weights into the format that can be used in `detrex`. Here's the detailed tutorial about the usage the conversion script.

### Download Pretrained Weights
`Torchvision 0.11.0` was released packed with better pretrained weights on numerous models including `ResNet`. More details can be found in [How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), here we collected the download scripts for TorchVision `ResNet` models.

<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Download</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<!-- TABLE BODY -->
 <tr><td align="left">ResNet-50 (ImageNet1k-V1) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O r50_v1.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">76.130</td>
<td align="center">92.862</td>
</tr>
 <tr><td align="left"> ResNet-50 (ImageNet1k-V2) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth -O r50_v2.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">80.858</td>
<td align="center">95.434</td>
</tr>
 <tr><td align="left"> ResNet-101 (ImageNet1k-V1) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet101-63fe2227.pth -O r101_v1.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">77.374</td>
<td align="center">93.546</td>
</tr>
 <tr><td align="left"> ResNet-101 (ImageNet1k-V2) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet101-cd907fc2.pth -O r101_v2.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">81.886</td>
<td align="center">95.780</td>
</tr>
 <tr><td align="left"> ResNet-152 (ImageNet1k-V1) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet152-394f9c45.pth -O r152_v1.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">78.312</td>
<td align="center">94.046</td>
</tr>
 <tr><td align="left"> ResNet-152 (ImageNet1k-V2) </td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet152-f82ba261.pth -O r152_v2.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">82.284</td>
<td align="center">96.002</td>
</tr>
</tbody></table>

**Note:** `ImageNet1k-V1` means the old pretrained weights. `ImageNet1k-V2` means the improved baseline results.

### Run the Conversion

<details>
<summary> <b> convert-torchvision-to-d2 (borrowed from detectron2) </b> </summary>

```python
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl
  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"
  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
```

</details>
<p></p>

Firstly, create `convert-torchvision-to-d2.py` and copy the relative code mentioned above, then run:

```bash
python convert-torchvision-to-d2.py \
    /path/to/r101_v1.pth \  # path to the downloaded pretrained weights
    ./r101_v1.pkl  # where to save the converted weights
```

Then, change the training configs:
```bash
# your own config.py

train.init_checkpoint = "path/to/r101_v1.pkl"

# make sure that the model config is consistent 
# with the following settings
model.backbone.stages.depth = 101
model.pixel_mean = [123.675, 116.280, 103.530]
model.pixel_std = [58.395, 57.120, 57.375]
```


## Convert DETRs Pretrained Models
These conversions are modified from the [detr-d2 conversion script](https://github.com/facebookresearch/detr/blob/main/d2/converter.py) to convert models trained by the original repo into the format of detrex models.

### Convert DETR
<details>
<summary> <b> convert-detr.py </b> </summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of DETR to be used with the detrex version.
"""
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex detr model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add new convert content
        if "encoder.layers" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")

        if "decoder" in k:
            if "decoder.norm" in k:
                k = k.replace("decoder.norm", "decoder.post_norm_layer")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            elif "multihead_attn" in k:
                k = k.replace("multihead_attn", "attentions.1.attn")

        # old fashion of detr convert function
        # k = "detr." + k
        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 92:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
```
</details>
<p></p>

Run the following script:
```bash
python convert-detr.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
                       --output_model converted_detr_model.pth
```

### Convert Deformable-DETR
<details>
<summary> <b> convert-deformable-detr.py </b> </summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of Deformable-DETR to be used with the detrex version.
"""
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex deformable-detr model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add input_proj converter
        if "input_proj" in k:
            if "input_proj.0.0" in k:
                k = k.replace("input_proj.0.0", "neck.convs.0.conv")
            if "input_proj.0.1" in k:
                k = k.replace("input_proj.0.1", "neck.convs.0.norm")
            if "input_proj.1.0" in k:
                k = k.replace("input_proj.1.0", "neck.convs.1.conv")
            if "input_proj.1.1" in k:
                k = k.replace("input_proj.1.1", "neck.convs.1.norm")
            if "input_proj.2.0" in k:
                k = k.replace("input_proj.2.0", "neck.convs.2.conv")
            if "input_proj.2.1" in k:
                k = k.replace("input_proj.2.1", "neck.convs.2.norm")
            if "input_proj.3.0" in k:
                k = k.replace("input_proj.3.0", "neck.extra_convs.0.conv")
            if "input_proj.3.1" in k:
                k = k.replace("input_proj.3.1", "neck.extra_convs.0.norm")

        # add new convert content
        if "encoder.layers" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")

        if "decoder" in k:
            if "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.1")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.0")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            elif "cross_attn" in k:
                k = k.replace("cross_attn", "attentions.1")

        if "level_embed" in k:
            k = k.replace("level_embed", "level_embeds")

        if "query_embed" in k:
            k = k.replace("query_embed", "query_embedding")

        # k = "detr." + k
        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 91:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
```
</details>
<p></p>

Firstly, download the pretrained model from [Deformable-DETR Main Results](https://github.com/fundamentalvision/Deformable-DETR#main-results).

Then run:
```bash
python convert-deformable-detr.py --source_model path/to/pretrained_weight.pth \
                                  --output_model converted_deformable_detr_model.pth
```


### Convert DAB-DETR
<details>
<summary> <b> convert-dab-detr.py </b> </summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of DAB-DETR to be used with the detrex version.
"""
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add new convert content
        if "decoder" in k:
            if "decoder.norm" in k:
                k = k.replace("decoder.norm", "decoder.post_norm_layer")
            if "ca_kcontent_proj" in k:
                k = k.replace("ca_kcontent_proj", "attentions.1.key_content_proj")
            elif "ca_kpos_proj" in k:
                k = k.replace("ca_kpos_proj", "attentions.1.key_pos_proj")
            elif "ca_qcontent_proj" in k:
                k = k.replace("ca_qcontent_proj", "attentions.1.query_content_proj")
            elif "ca_qpos_proj" in k:
                k = k.replace("ca_qpos_proj", "attentions.1.query_pos_proj")
            elif "ca_qpos_sine_proj" in k:
                k = k.replace("ca_qpos_sine_proj", "attentions.1.query_pos_sine_proj")
            elif "ca_v_proj" in k:
                k = k.replace("ca_v_proj", "attentions.1.value_proj")
            elif "sa_kcontent_proj" in k:
                k = k.replace("sa_kcontent_proj", "attentions.0.key_content_proj")
            elif "sa_kpos_proj" in k:
                k = k.replace("sa_kpos_proj", "attentions.0.key_pos_proj")
            elif "sa_qcontent_proj" in k:
                k = k.replace("sa_qcontent_proj", "attentions.0.query_content_proj")
            elif "sa_qpos_proj" in k:
                k = k.replace("sa_qpos_proj", "attentions.0.query_pos_proj")
            elif "sa_v_proj" in k:
                k = k.replace("sa_v_proj", "attentions.0.value_proj")
            elif "self_attn.out_proj" in k:
                k = k.replace("self_attn.out_proj", "attentions.0.out_proj")
            elif "cross_attn.out_proj" in k:
                k = k.replace("cross_attn.out_proj", "attentions.1.out_proj")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "activation" in k:
                k = k.replace("activation", "ffns.0.layers.0.1")

        if "encoder" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            if "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")
            elif "activation" in k:
                k = k.replace("activation", "ffns.0.layers.0.1")

        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 91:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
```

</details>
<p></p>

Firstly, download the pretrained model from [DAB-DETR Model Zoo](https://github.com/IDEA-opensource/DAB-DETR#model-zoo).

Then run:
```bash
python convert-dab-detr.py --source_model path/to/pretrained_weight.pth \
                           --output_model converted_dab_detr_model.pth
```

### Convert DAB-Deformable-DETR
<details>
<summary> <b> convert-dab-deformable-detr.py </b> </summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of DAB-Deformable-DETR to be used with the detrex version.
"""
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add input_proj converter
        if "input_proj" in k:
            if "input_proj.0.0" in k:
                k = k.replace("input_proj.0.0", "neck.convs.0.conv")
            if "input_proj.0.1" in k:
                k = k.replace("input_proj.0.1", "neck.convs.0.norm")
            if "input_proj.1.0" in k:
                k = k.replace("input_proj.1.0", "neck.convs.1.conv")
            if "input_proj.1.1" in k:
                k = k.replace("input_proj.1.1", "neck.convs.1.norm")
            if "input_proj.2.0" in k:
                k = k.replace("input_proj.2.0", "neck.convs.2.conv")
            if "input_proj.2.1" in k:
                k = k.replace("input_proj.2.1", "neck.convs.2.norm")
            if "input_proj.3.0" in k:
                k = k.replace("input_proj.3.0", "neck.extra_convs.0.conv")
            if "input_proj.3.1" in k:
                k = k.replace("input_proj.3.1", "neck.extra_convs.0.norm")

        # add new convert content
        if "encoder.layers" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")

        if "decoder" in k:
            if "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.1")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.0")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            elif "cross_attn" in k:
                k = k.replace("cross_attn", "attentions.1")

        if "level_embed" in k:
            k = k.replace("level_embed", "level_embeds")

        # k = "detr." + k
        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 91:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
```

</details>
<p></p>

Firstly, download the pretrained model from [DAB-DETR Model Zoo](https://github.com/IDEA-opensource/DAB-DETR#model-zoo).

Then run:
```bash
python convert-dab-deformable-detr.py --source_model path/to/pretrained_weight.pth \
                                      --output_model converted_dab_deformable_detr_model.pth
```

### Convert DN-DETR
<details>
<summary> <b> convert-dn-detr.py </b> </summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of DN-DETR to be used with the detrex version.
"""
import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("detrex model converter")

    parser.add_argument(
        "--source_model", default="", type=str, help="Path or url to the DETR model to convert"
    )
    parser.add_argument(
        "--output_model", default="", type=str, help="Path where to save the converted model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    label_enc_coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.source_model, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone." + k

        # add new convert content
        if "label_enc" in k:
            k = k.replace("label_enc", "label_encoder")
        if "decoder" in k:
            if "decoder.norm" in k:
                k = k.replace("decoder.norm", "decoder.post_norm_layer")
            if "ca_kcontent_proj" in k:
                k = k.replace("ca_kcontent_proj", "attentions.1.key_content_proj")
            elif "ca_kpos_proj" in k:
                k = k.replace("ca_kpos_proj", "attentions.1.key_pos_proj")
            elif "ca_qcontent_proj" in k:
                k = k.replace("ca_qcontent_proj", "attentions.1.query_content_proj")
            elif "ca_qpos_proj" in k:
                k = k.replace("ca_qpos_proj", "attentions.1.query_pos_proj")
            elif "ca_qpos_sine_proj" in k:
                k = k.replace("ca_qpos_sine_proj", "attentions.1.query_pos_sine_proj")
            elif "ca_v_proj" in k:
                k = k.replace("ca_v_proj", "attentions.1.value_proj")
            elif "sa_kcontent_proj" in k:
                k = k.replace("sa_kcontent_proj", "attentions.0.key_content_proj")
            elif "sa_kpos_proj" in k:
                k = k.replace("sa_kpos_proj", "attentions.0.key_pos_proj")
            elif "sa_qcontent_proj" in k:
                k = k.replace("sa_qcontent_proj", "attentions.0.query_content_proj")
            elif "sa_qpos_proj" in k:
                k = k.replace("sa_qpos_proj", "attentions.0.query_pos_proj")
            elif "sa_v_proj" in k:
                k = k.replace("sa_v_proj", "attentions.0.value_proj")
            elif "self_attn.out_proj" in k:
                k = k.replace("self_attn.out_proj", "attentions.0.out_proj")
            elif "cross_attn.out_proj" in k:
                k = k.replace("cross_attn.out_proj", "attentions.1.out_proj")
            elif "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")
            elif "norm3" in k:
                k = k.replace("norm3", "norms.2")
            elif "activation" in k:
                k = k.replace("activation", "ffns.0.layers.0.1")

        if "encoder" in k:
            if "self_attn" in k:
                k = k.replace("self_attn", "attentions.0.attn")
            if "linear1" in k:
                k = k.replace("linear1", "ffns.0.layers.0.0")
            elif "linear2" in k:
                k = k.replace("linear2", "ffns.0.layers.1")
            elif "norm1" in k:
                k = k.replace("norm1", "norms.0")
            elif "norm2" in k:
                k = k.replace("norm2", "norms.1")
            elif "activation" in k:
                k = k.replace("activation", "ffns.0.layers.0.1")

        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 91:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print(
                    "Head conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue

        if "label_enc" in old_k:
            v = model_to_convert[old_k].detach()
            print("Label enc conversion:", v.shape[0])
            if v.shape[0] == 92:
                shape_old = v.shape
                model_converted[k] = v[label_enc_coco_idx]
                print(
                    "Label enc conversion: changing shape from {} to {}".format(
                        shape_old, model_converted[k].shape
                    )
                )
                continue

        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
```

</details>
<p></p>

Firstly, download the pretrained model from [DN-DETR Model Zoo](https://github.com/IDEA-opensource/DN-DETR#model-zoo).

Then run:
```bash
python convert-dn-detr.py --source_model path/to/pretrained_weight.pth \
                          --output_model converted_dn_detr_model.pth
```

### Convert DN-Deformable-DETR
<details>
<summary> <b> convert-deformable-detr.py </b> </summary>

</details>
<p></p>

### Convert DINO
<details>
<summary> <b> convert-deformable-detr.py </b> </summary>

</details>
<p></p>
