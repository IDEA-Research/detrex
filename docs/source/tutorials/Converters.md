# Convert Pretrained Models
This document provides a brief intro of how to convert the pretrained model into the format of detrex.


## Convert TorchVision Pretrained ResNet Models
To use the detectron2 provided pretrained weights, please refer to [ImageNet Pretrained Models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#imagenet-pretrained-models). Here we've noticed that detectron2 only provided a converted torchvision `ResNet-50` model. For more pretrained models like `ResNet{101, 152}`. You can use the detectron2 provided [conversion script](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py) to convert the torchvision pretrained weights into the format that can be used in `detrex`. Here's the detailed tutorial about the usage the conversion script.

### 1. Download Pretrained Weights
`Torchvision 0.11.0` was released packed with better pretrained weights on numerous models including `ResNet`. More details can be found in [How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), here we collected the download link for `TorchVision ResNet` models.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Download</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a>ResNet-50 (ImageNet1k-V1) </a></td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet50-0676ba61.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">76.130</td>
<td align="center">92.862</td>
</tr>
 <tr><td align="left"><a>ResNet-50 (ImageNet1k-V2) </a></td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">80.858</td>
<td align="center">95.434</td>
</tr>
 <tr><td align="left"><a>ResNet-101 (ImageNet1k-V1) </a></td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet101-63fe2227.pth </code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">77.374</td>
<td align="center">93.546</td>
</tr>
 <tr><td align="left"><a>ResNet-101 (ImageNet1k-V2) </a></td>
<td align="center"> <details><summary> script </summary><pre><code> wget https://download.pytorch.org/models/resnet101-cd907fc2.pth</code></pre></details> </td>
<td align="center">IN1k</td>
<td align="center">81.886</td>
<td align="center">95.780</td>
</tr>
</tbody></table>

**Note:** `ImageNet1k-V1` means the old pretrained weights. `ImageNet1k-V2` means the improved baseline results.