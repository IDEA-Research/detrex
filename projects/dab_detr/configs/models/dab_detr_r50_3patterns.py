from .dab_detr_r50 import model


# using 3 pattern embeddings as in Anchor-DETR
model.transformer.num_patterns = 3
