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
We also provides converters for a partial of projects in detrex. These conversions are modified from the [detr-d2 conversion script](https://github.com/facebookresearch/detr/blob/main/d2/converter.py) to convert models trained by the original repo into the format of detrex models.

- converter for DETR: [convert_detr_to_detrex](https://github.com/IDEA-Research/detrex/blob/main/projects/detr/converter.py)
- converter for Deformable-DETR: [convert_deformable_detr_to_detrex](https://github.com/IDEA-Research/detrex/blob/main/projects/deformable_detr/converter.py)
- converter for ConditionalDETR: [convert_conditional_detr_to_detrex](https://github.com/IDEA-Research/detrex/blob/main/projects/conditional_detr/converter.py)
- converter for DN-Deformable-DETR: [convert_dn_deformable_detr_to_detrex](https://github.com/IDEA-Research/detrex/blob/main/projects/dn_deformable_detr/converter.py)

All these converters can be runned as:
```python
python converter.py --source_model /path/to/pretrained_weight.pth --output_model converted_model.pth
```

