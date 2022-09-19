detrex.modeling
##############################

backbone
------------------------------
.. currentmodule:: detrex.modeling
.. automodule:: detrex.modeling.backbone
    :member-order: bysource
    :members:
        ResNet,
        make_stage,
        ConvNeXt,
        FocalNet,
        TimmBackbone,
        TorchvisionBackbone,

neck
------------------------------
.. currentmodule:: detrex.modeling
.. automodule:: detrex.modeling.neck
    :member-order: bysource
    :members:
        ChannelMapper,


matcher
------------------------------
.. currentmodule:: detrex.modeling
.. automodule:: detrex.modeling.matcher
    :member-order: bysource
    :members:
        HungarianMatcher,


losses
------------------------------
.. currentmodule:: detrex.modeling
.. automodule:: detrex.modeling.losses
    :member-order: bysource
    :members:
        sigmoid_focal_loss,
        dice_loss,