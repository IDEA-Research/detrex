# Frequently Asked Questions
Here we provided some common troubles faced by the users and their corresponding solutions here. If the contents here do not cover your issue, please create an issue about it.

We've already opened an issue about FAQs, please refer to [Frequently Asked Questions](https://github.com/IDEA-Research/detrex/issues/109) for more details.

## Installation

<details>
<summary> <b> ImportError: Cannot import 'detrex._C', therefore 'MultiscaleDeformableAttention' is not available </b> </summary>

detrex need **CUDA runtime** to build the `MultiScaleDeformableAttention` operator. In most cases, users do not need to specify this environment variable if you have installed cuda correctly. The default path of CUDA runtime is `usr/local/cuda`. If you find your `CUDA_HOME` is `None`.  You may solve it as follows:
- If you've already installed **CUDA runtime** in your environments, specify the environment variable (here we take cuda-11.3 as an example):
```bash
export CUDA_HOME=/path/to/cuda-11.3/
```
- If you do not find the CUDA runtime in your environments, consider install it following the [CUDA Toolkit Installation](https://developer.nvidia.com/cuda-toolkit) to install CUDA. Then specify the environment variable `CUDA_HOME`.
- After setting `CUDA_HOME`, rebuild detrex again by running `pip install -e .`

You can also refer to these issues for more details: [#98](https://github.com/IDEA-Research/detrex/issues/98), [#85](https://github.com/IDEA-Research/detrex/issues/85).
</details>

## Training

<details>
<summary> <b> assert (boxes1[:, 2:] >= boxes1[:, :2]).all()" in `generalized_box_iou` </b> </summary>

This means the model produces **illegal box predictions**. You may solute this issue as follows:
    1. Check the learning rate which should not be too large. The DETR-like models are usually trained using `AdamW` with `lr=1e-4`.
    2. Make sure that your model are **initilized correctly**. Please check the `init_weights()` function in models.

</details>

<details>
<summary> <b> How to not filter empty annotations during training?  </b> </summary>

There're few ways for you to **not filter empty annotations** during training.

1. modify configs in [configs/common/data/coco_detr.py](https://github.com/IDEA-Research/detrex/blob/5d866bd115b6e0e6a0eac253761855196615e5c4/configs/common/data/coco_detr.py#L17) as follows:
```python
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train", filter_empty=False),
    ...,
)
```

1. modify configs in projects as [dino_r50_4scale_24ep.py](https://github.com/IDEA-Research/detrex/blob/5d866bd115b6e0e6a0eac253761855196615e5c4/projects/dino/configs/dino_r50_4scale_24ep.py#L48).

```python
# your config.py
dataloader = get_config("common/data/coco_detr.py").dataloader

# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = False
```

3. modify your training scripts to override the config.
```python
cd detrex
python tools/train_net.py --config-file projects/dino/configs/path/to/config.py --num-gpus 8 dataloader.train.dataset.filter_empy=False
```

You can also refer to these issues for more details: [#issue78-comments](https://github.com/IDEA-Research/detrex/issues/78#issuecomment-1284054108)

</details>


<details>
<summary> <b> RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:54980 (errno: 98 - Address already in use). </b> </summary>

This means that the process you started earlier did not exit correctly, there's two solutions:

1. kill the process you started before totally
2. change the running port by setting `--dist-url`

```bash
python tools/train_net.py \
    --config-file path/to/config.py \
    --num-gpus 8 \
    --dist-url tcp://127.0.0.1:12345 \
```

</details>

<details>
<summary> <b> How to inference DINO on CPU?</b> </summary>

Please refer to this PR [#157](https://github.com/IDEA-Research/detrex/issues/157)  for more details

</details>

<details>
<summary> <b> How to train the custom coco-like format dataset?</b> </summary>

Please refer to this PR [#186](https://github.com/IDEA-Research/detrex/issues/186)  for more details

</details>

<!-- <details>
<summary> <b> </b> </summary>



</details> -->