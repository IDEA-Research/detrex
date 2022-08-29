libai.models
##############################
Supported models in LiBai(李白)

- `VisionTransformer`_
- `SwinTransformer`_
- `ResMLP`_
- `BERT`_
- `RoBERTa`_
- `T5`_
- `GPT-2`_

.. _VisionTransformer: https://arxiv.org/abs/2010.11929
.. _SwinTransformer: https://arxiv.org/abs/2103.14030
.. _ResMLP: https://arxiv.org/abs/2105.03404
.. _BERT: https://arxiv.org/abs/1810.04805
.. _RoBERTa: https://arxiv.org/abs/1907.11692
.. _T5: https://arxiv.org/abs/1910.10683
.. _GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf



.. currentmodule:: libai.models

.. automodule:: libai.models.build
    :members:
        build_model,
        build_graph,

VisionTransformer
-----------------
.. automodule:: libai.models.vision_transformer
    :members: 
        VisionTransformer,

SwinTransformer
---------------
.. automodule:: libai.models.swin_transformer
    :members: 
        SwinTransformer,

ResMLP
-------
.. automodule:: libai.models.resmlp
    :members:
        ResMLP,

BERT
----
.. automodule:: libai.models.bert_model
    :members: 
        BertModel,
        BertForPreTraining,

RoBERTa
-------
.. automodule:: libai.models.roberta_model
    :members: 
        RobertaModel,
        RobertaForPreTraining,
        RobertaForCausalLM,

T5
---
.. automodule:: libai.models.t5_model
    :members: 
        T5ForPreTraining,
        T5Model,

GPT-2
-----
.. automodule:: libai.models.gpt_model
    :members: 
        GPTForPreTraining,
        GPTModel