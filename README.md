# detrex
IDEA open source toolbox for visual recognition tasks


## Environments Setting
- install `detectron2 == 0.6.0`

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

- install `detrex`

```bash
git clone https://github.com/rentainhe/IDEADet
cd IDEADet
pip install -r requirements.txt
pip install -e .
```

## Datasets Preparation
- `COCO 2017`

Prepare the datasets as follows:
```bash
datasets/
    coco/
    annotations/
        instances_{train,val}2017.json
        person_keypoints_{train,val}2017.json
    {train,val}2017/
        # image files that are mentioned in the corresponding json
```

Set the datasets path:
```bash
export DETECTRON2_DATASETS="path/to/datasets/"
```

Use the prepared datasets in `dgx061`:
```bash
export DETECTRON2_DATASETS=/comp_robot/rentianhe/code/IDEADet/datasets
```

## Training DAB-DETR
```bash
cd projects/dab_detr
python train_net.py --config-file configs/dab_detr_r50_50epoch.py --num-gpus 1
```

## Evaluate DAB-DETR
```bash
cd projects/dab_detr
python train_net.py --config-file configs/dab_detr_r50_50epoch.py --num-gpus 1 --eval-only
```
