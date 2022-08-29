## Changelog

### v0.2.0 (07/07/2022)
**New Features:**
- Support evaluation enabled and set `eval_iter`
- Support customized sampler in `config.py`
- Support rdma for pipeline-model-parallel
- Support multi fused kernel 
   - fused_scale_mask_softmax_dropout
   - fused_scale_tril_softmax_mask_scale
   - fused_self_attention in branch `libai_bench`
- User Experience Optimization
- Optimization for training throughput, see [benchmark](https://libai.readthedocs.io/en/latest/tutorials/get_started/Benchmark.html) for more details

**New Supported Models:**

- Support 3D parallel [Roberta](https://arxiv.org/abs/1907.11692) model
- Support 2D parallel (data parallel + tensor model parallel) [SimCSE](https://arxiv.org/abs/2104.08821) model
- Support Data parallel [MAE](https://arxiv.org/abs/2111.06377) model
- Support Data parallel [MOCOV3](https://arxiv.org/abs/2104.02057) model

### v0.1.0 (22/03/2022)

**New Features:**
- Support Data Parallelism
- Support 1D Tensor Parallelism
- Support Pipeline Parallelism
- Unified distributed Layers for both single-GPU and multi-GPU training
- `LazyConfig` system for more flexible syntax and no predefined structures
- Easy-to-use trainer and engine
- Support both CV and NLP data processing
- Mixed Precision Training
- Activation Checkpointing
- Gradient Accumulation
- Gradient Clipping
- Zero Redundancy Optimizer (ZeRO)

**Supported Models:**
- Support 3D parallel [BERT](https://arxiv.org/abs/1810.04805) model
- Support 3D parallel [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model
- Support 3D parallel [T5](https://arxiv.org/abs/1910.10683) model
- Support 3D parallel [Vision Transformer](https://arxiv.org/abs/2010.11929) model
- Support Data parallel [Swin Transformer](https://arxiv.org/abs/2103.14030) model
- Support finetune task in [QQP project](https://github.com/Oneflow-Inc/libai/tree/main/projects/QQP)
- Support text classification task in [text classification project](https://github.com/Oneflow-Inc/libai/tree/main/projects/text_classification)

