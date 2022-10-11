# Model Zoo

## Common Settings
- All COCO models were trained on `coco_2017_train` and evaluated on `coco_2017_val`.
- All models were trained using distributed training.
- Most models were trained with 50 epochs settings (~51 COCO epochs) with multi-step LR scheduler which is the common setting in DETR-like methods.


## COCO Object Detection Baselines
Here we provides our pretrained baselines in `detrex beta v0.1.0`. And more pretrained weights will be released in the future version.

### Deformable-DETR
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/deformable_detr/configs/deformable_detr_r50_with_box_refinement_50ep.py"> Deformable-DETR + Box Refinement </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.32</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/deformable_detr_with_box_refinement_50ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/deformable_detr/configs/deformable_detr_r50_two_stage_50ep.py"> Deformable-DETR + Box Refinement + Two Stage </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.28</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_r50_two_stage_50ep.pth"> model </a></td>
</tr>
</tbody></table>

### Conditional-DETR
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: conditional_detr_r50_50ep -->
 <tr><td align="left"><a href="https://github.com/IDEA-Research/detrex/blob/main/projects/conditional_detr/configs/conditional_detr_r50_50ep.py">Conditional-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">41.60</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/conditional_detr_r50_50ep.pth">model</a></td>
</tr>
</tbody></table>

### DAB-DETR
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dab_detr/configs/dab_detr_r50_50ep.py"> DAB-DETR-R50 </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">43.28</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_r50_50ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dab_detr/configs/dab_detr_r101_50ep.py"> DAB-DETR-R101 </a> </td>
<td align="center">R101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">43.98</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_r101_50ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dab_detr/configs/dab_detr_swin_t_in1k_50ep.py"> DAB-DETR-Swin-T </a> </td>
<td align="center">Swin-T</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.17</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_swin_t_in1k_50ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dab_deformable_detr/configs/dab_deformable_detr_r50_50ep.py"> DAB-Deformable-DETR-R50 </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">48.87</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_deformable_r50_50ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dab_deformable_detr/configs/dab_deformable_detr_r50_two_stage_50ep.py"> DAB-Deformable-DETR-R50-Two-Stage </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">49.54</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_deformable_r50_two_stage_50ep.pth"> model </a></td>
</tr>
</tbody></table>


### DN-DETR
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dn_detr/configs/dn_detr_r50_50ep.py"> DN-DETR-R50 </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">44.73</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dn_detr_r50_50ep.pth"> model </a></td>
</tr>
</tbody></table>

### DINO
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r50_4scale_12ep.py"> DINO-R50-4Scale </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">49.05</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dino_r50_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r101_4scale_12ep.py"> DINO-R101-4Scale </a> </td>
<td align="center">R101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">49.95</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r101_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_tiny_224_4scale_12ep.py"> DINO-Swin-T-224-4scale </a> </td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">51.30</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_tiny_224_4scale_12ep.py"> DINO-Swin-T-224-4scale </a> </td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">52.50</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_small_224_4scale_12ep.py"> DINO-Swin-S-224-4scale </a> </td>
<td align="center">Swin-Small-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">52.96</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_small_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_base_384_4scale_12ep.py"> DINO-Swin-B-384-4scale </a> </td>
<td align="center">Swin-Base-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">55.83</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_224_4scale_12ep.py"> DINO-Swin-L-224-4scale </a> </td>
<td align="center">Swin-Large-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">56.92</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_384_4scale_12ep.py"> DINO-Swin-L-384-4scale </a> </td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">56.93</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_4scale_12ep.pth"> model </a></td>
</tr>
</tbody></table>
