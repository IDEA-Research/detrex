# Model Zoo

## Common Settings
- All COCO models were trained on `coco_2017_train` and evaluated on `coco_2017_val`.
- All models were trained using distributed training.
- Most models were trained with 50 epochs settings (~51 COCO epochs) with multi-step LR scheduler which is the common setting in DETR-like methods.


## COCO Object Detection Baselines
Here we provides our pretrained baselines with **detrex**. And more pretrained weights will be released in the future version.

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
<td align="center">46.99</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_with_box_refinement_50ep_new.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/deformable_detr/configs/deformable_detr_r50_two_stage_50ep.py"> Deformable-DETR + Box Refinement + Two Stage </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">48.19</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_r50_two_stage_50ep_new.pth"> model </a></td>
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
<td align="center">Swin-Tiny-224</td>
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
<th valign="bottom">Denoising Queries</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r50_4scale_12ep.py"> DINO-R50-4Scale </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.05</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dino_r50_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r50_4scale_12ep_300dn.py"> DINO-R50-4Scale </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">300</td>
<td align="center">49.45</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_r50_4scale_12ep_300dn.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r50_4scale_24ep.py"> DINO-R50-4Scale </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">24</td>
<td align="center">100</td>
<td align="center">50.60</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r50_4scale_24ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_r101_4scale_12ep.py"> DINO-R101-4Scale </a> </td>
<td align="center">R101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">49.95</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r101_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_tiny_224_4scale_12ep.py"> DINO-Swin-T-224-4Scale </a> </td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">51.30</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_tiny_224_4scale_12ep.py"> DINO-Swin-T-224-4Scale </a> </td>
<td align="center">Swin-Tiny-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.50</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_small_224_4scale_12ep.py"> DINO-Swin-S-224-4Scale </a> </td>
<td align="center">Swin-Small-224</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">52.96</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_small_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_base_384_4scale_12ep.py"> DINO-Swin-B-384-4Scale </a> </td>
<td align="center">Swin-Base-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">55.83</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_224_4scale_12ep.py"> DINO-Swin-L-224-4Scale </a> </td>
<td align="center">Swin-Large-224</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">56.92</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_224_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_384_4scale_12ep.py"> DINO-Swin-L-384-4Scale </a> </td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">56.93</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_4scale_12ep.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/dino_swin_large_384_4scale_36ep.py"> DINO-Swin-L-384-4Scale </a> </td>
<td align="center">Swin-Large-384</td>
<td align="center">IN22k to IN1k</td>
<td align="center">12</td>
<td align="center">100</td>
<td align="center">58.09</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_swin_large_384_4scale_36ep.pth"> model </a></td>
</tr>
</tbody></table>


### H-Deformable-DETR
<table class="docutils"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrained</th>
<th valign="bottom">Query</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_r50_two_stage_12ep.py"> H-Deformable-DETR-R50 + tricks </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">48.9</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_r50_two_stage_36ep.py"> H-Deformable-DETR-R50 + tricks </a> </td>
<td align="center">R50</td>
<td align="center">IN1k</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">50.3</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_tiny_two_stage_12ep.py"> H-Deformable-DETR-Swin-T + tricks </a> </td>
<td align="center">Swin-Tiny</td>
<td align="center">IN1k</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">50.6</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_tiny_two_stage_36ep.py"> H-Deformable-DETR-Swin-T + tricks </a> </td>
<td align="center">Swin-Tiny</td>
<td align="center">IN1k</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">53.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href=""> H-Deformable-DETR-Swin-L + tricks </a> </td>
<td align="center">Swin-Large</td>
<td align="center">IN22k</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">56.2</td>
<td align="center"> <a href=""> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_12ep.py"> H-Deformable-DETR-Swin-L + tricks </a> </td>
<td align="center">Swin-Large</td>
<td align="center">IN22k</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">56.2</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_36ep.py"> H-Deformable-DETR-Swin-L + tricks </a> </td>
<td align="center">Swin-Large</td>
<td align="center">IN22k</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">57.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_12ep_900queries.py"> H-Deformable-DETR-Swin-L + tricks </a> </td>
<td align="center">Swin-Large</td>
<td align="center">IN22k</td>
<td align="center">900</td>
<td align="center">12</td>
<td align="center">56.4</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"> model </a></td>
</tr>
 <tr><td align="left"> <a href="https://github.com/IDEA-Research/detrex/blob/main/projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_36ep_900queries.py"> H-Deformable-DETR-Swin-L + tricks </a> </td>
<td align="center">Swin-Large</td>
<td align="center">IN22k</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">57.5</td>
<td align="center"> <a href="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.2/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"> model </a></td>
</tr>
</tbody></table>
