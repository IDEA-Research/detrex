# Model Zoo

## Common Settings
- All COCO models were trained on `coco_2017_train` and evaluated on `coco_2017_val`.
- All models were trained using distributed training.
- Most models were trained with 50 epochs settings (~51 COCO epochs) with multi-step LR scheduler which is the common setting in DETR-like methods.


## COCO Object Detection Baselines
Here we provides our pretrained baselines in `detrex beta v0.1.0`. And more pretrained weights will be released in the future version.

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
<td align="center">50</td>
<td align="center">49.05</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dino_r50_4scale_12ep.pth"> model </a></td>
</tr>
</tbody></table>
