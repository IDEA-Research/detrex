# Download Pretrained Backbone Weights

Here we collect the **links** of the backbone models which makes it easier for users to **download pretrained weights** for the **builtin backbones**. And this document will be kept updated. Most included models are borrowed from their original sources. Many thanks for their nicely work in the backbone area.

## ResNet
We've already provided the tutorials of **using torchvision pretrained ResNet models** here: [Download TorchVision ResNet Models](https://detrex.readthedocs.io/en/latest/tutorials/Converters.html#download-pretrained-weights).

## Swin-Transformer
Here we borrowed the download links from the [official implementation](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models) of Swin-Transformer.

### Swin-Tiny
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<th valign="bottom">22K Model</th>
<th valign="bottom">1K Model</th>
<!-- TABLE BODY -->
 <tr><td align="left"> Swin-Tiny </td>
<td align="center">ImageNet-1K</td>
<td align="center">224x224</td>
<td align="center">81.2</td>
<td align="center">95.5</td>
<td align="center"> - </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth">download</a> </td>
</tr>
 <tr><td align="left"> Swin-Tiny </td>
<td align="center">ImageNet-22K</td>
<td align="center">224x224</td>
<td align="center">80.9</td>
<td align="center">96.0</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"> download </a> </td>
</tr>
</tbody></table>

<details open>
<summary> <b> Using Swin-Tiny Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.1,
    window_size=7,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
# train.init_checkpoint = "/path/to/swin_tiny_patch4_window7_224.pth"
train.init_checkpoint = "/path/to/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
```

</details>

### Swin-Small
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<th valign="bottom">22K Model</th>
<th valign="bottom">1K Model</th>
<!-- TABLE BODY -->
 <tr><td align="left"> Swin-Small </td>
<td align="center">ImageNet-1K</td>
<td align="center">224x224</td>
<td align="center">83.2</td>
<td align="center">96.2</td>
<td align="center"> - </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth">download</a> </td>
</tr>
 <tr><td align="left"> Swin-Small </td>
<td align="center">ImageNet-22K</td>
<td align="center">224x224</td>
<td align="center">83.2</td>
<td align="center">97.0</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth"> download </a> </td>
</tr>
</tbody></table>

<details open>
<summary> <b> Using Swin-Small Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=96,
    depths=(2, 2, 18, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.2,
    window_size=7,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
# train.init_checkpoint = "/path/to/swin_small_patch4_window7_224.pth"
train.init_checkpoint = "/path/to/swin_small_patch4_window7_224_22kto1k_finetune.pth"
```

</details>

### Swin-Base
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<th valign="bottom">22K Model</th>
<th valign="bottom">1K Model</th>
<!-- TABLE BODY -->
 <tr><td align="left"> Swin-Base </td>
<td align="center">ImageNet-1K</td>
<td align="center">224x224</td>
<td align="center">83.5</td>
<td align="center">96.5</td>
<td align="center"> - </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth">download</a> </td>
</tr>
 <tr><td align="left"> Swin-Base </td>
<td align="center">ImageNet-1K</td>
<td align="center">384x384</td>
<td align="center">84.5</td>
<td align="center">97.0</td>
<td align="center"> - </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth">download</a> </td>
</tr>
 <tr><td align="left"> Swin-Base </td>
<td align="center">ImageNet-22K</td>
<td align="center">224x224</td>
<td align="center">85.2</td>
<td align="center">97.5</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth"> download </a> </td>
</tr>
 <tr><td align="left"> Swin-Base </td>
<td align="center">ImageNet-22K</td>
<td align="center">384x384</td>
<td align="center">86.4</td>
<td align="center">98.0</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth"> download </a> </td>
</tr>
</tbody></table>

<details open>
<summary> <b> Using Swin-Base-224 Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=7,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
# train.init_checkpoint = "/path/to/swin_base_patch4_window7_224.pth"
train.init_checkpoint = "/path/to/swin_base_patch4_window7_224_22kto1k.pth"
```

<details open>
<summary> <b> Using Swin-Base-384 Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=12,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
# train.init_checkpoint = "/path/to/swin_base_patch4_window12_384.pth"
train.init_checkpoint = "/path/to/swin_base_patch4_window12_384_22kto1k.pth"
```

</details>

### Swin-Large
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Resolution</th>
<th valign="bottom">Acc@1</th>
<th valign="bottom">Acc@5</th>
<th valign="bottom">22K Model</th>
<th valign="bottom">1K Model</th>
<!-- TABLE BODY -->
 <tr><td align="left"> Swin-Large </td>
<td align="center">ImageNet-22K</td>
<td align="center">224x224</td>
<td align="center">86.3</td>
<td align="center">97.9</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth">download</a> </td>
</tr>
 <tr><td align="left"> Swin-Large </td>
<td align="center">ImageNet-22K</td>
<td align="center">384x384</td>
<td align="center">87.3</td>
<td align="center">98.2</td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth">download</a> </td>
<td align="center"> <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth"> download </a> </td>
</tr>
</tbody></table>

<details open>
<summary> <b> Using Swin-Large-224 Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=224,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=7,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
train.init_checkpoint = "/path/to/swin_large_patch4_window7_224_22kto1k.pth"
```

</details>

<details open>
<summary> <b> Using Swin-Large-384 Backbone in Config </b> </summary>

```python
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(1, 2, 3),
)

# setup init checkpoint path
train.init_checkpoint = "/path/to/swin_large_patch4_window12_384_22kto1k.pth"
```
</details>
