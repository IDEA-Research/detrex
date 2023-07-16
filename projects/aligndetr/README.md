# AlignDETR
## Introduction
AlignDETR is a variant of DETR(DEtection with Transformer), with a simple IoU-Aware BCE loss and better performance! It aims to solve the issue of misalignment problem spotted in DETR's output.
For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2304.07527). 
<img src="assets/aligndetr.png" >

## Installation 
See [installation instructions](INSTALL.md) 
## Usage
Train Example
```bash
python tools/train_net.py --config-file  aligndetr/aligndetr_k=2_r50_4scale_12ep.py --num-gpus 8
```
Evaluation Example
```bash
python tools/train_net.py --config-file  aligndetr/aligndetr_k=2_r50_4scale_12ep.py --num-gpus 8 --eval train.init_checkpoint=/path/to/checkpoint
```
## Model Zoo

|Model|AP|AP50|AP75|APs|APm|APl|weights|
|----|----|----|----|----|----|----|----|
|[AlignDETR-R50-12ep](aligndetr/configs/aligndetr_k%3D2_r50_4scale_12ep.py)|50.3|67.9|54.8|34.1|53.5|65.1|[Google Drive](https://drive.google.com/file/d/12xSxD_Z9KI8bejSlO1td1XcZ-Ns00rTi/view?usp=share_link)|
|[AlignDETR-R50-24ep](aligndetr/configs/aligndetr_k%3D2_r50_4scale_24ep.py)|51.4|69.1|55.8|35.5|54.6|65.7|[Google Drive](https://drive.google.com/file/d/1zQYe78fDdCmK3nwbAWYLvGCdQCFsTqoX/view?usp=share_link)|

## Acknowlegements

Our code is based on [detrex](https://github.com/IDEA-Research/detrex) and [detectron2](https://github.com/facebookresearch/detectron2).
## Citation

If you are interested in our work and use our method in your research, please cite
```
@misc{cai2023aligndetr,
      title={Align-DETR: Improving DETR with Simple IoU-aware BCE loss}, 
      author={Zhi Cai and Songtao Liu and Guodong Wang and Zheng Ge and Xiangyu Zhang and Di Huang},
      year={2023},
      eprint={2304.07527},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## License
This project is released under the [Apache 2.0 license](LICENSE).
