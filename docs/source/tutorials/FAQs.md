# Frequently Asked Questions
Here we provided some common troubles faced by the users and their corresponding solutions here. If the contents here do not cover your issue, please create an issue about it.

## Training
- "assert (boxes1[:, 2:] >= boxes1[:, :2]).all()" in `generalized_box_iou`
    This means the model produces **illegal box predictions**. You may solute this issue as follows:
    1. Check the learning rate which should not be too large. The DETR-like models are usually trained using `AdamW` with `lr=1e-4`.
    2. Make sure that your model are initilized correctly. Please check the `init_weights()` function in models.