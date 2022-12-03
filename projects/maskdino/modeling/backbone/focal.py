# --------------------------------------------------------
# FocalNet for Semantic Segmentation
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang
# --------------------------------------------------------
import math
import time
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from util.misc import NestedTensor
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False, 
        use_postln_in_modulation=False, normalize_modulator=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, 
                 use_postln=False, use_postln_in_modulation=False, 
                 normalize_modulator=False, 
                 use_layerscale=False, 
                 layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop, 
            use_postln_in_modulation=use_postln_in_modulation, 
            normalize_modulator=normalize_modulator, 
        )            

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # FM
        x = self.modulation(x).view(B, H * W, C)
        if self.use_postln:
            x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)

        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_window=9, 
                 focal_level=2, 
                 use_conv_embed=False,     
                 use_postln=False,          
                 use_postln_in_modulation=False, 
                 normalize_modulator=False, 
                 use_layerscale=False,                   
                 use_checkpoint=False
        ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                focal_window=focal_window, 
                focal_level=focal_level, 
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator, 
                use_layerscale=use_layerscale, 
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2, 
                in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False
            )

        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_reshaped = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W)
            x_down = self.downsample(x_reshaped)      
            x_down = x_down.flatten(2).transpose(1, 2)            
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, use_conv_embed=False, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)                    
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=1600,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 focal_levels=[2,2,2,2], 
                 focal_windows=[9,9,9,9],
                 use_conv_embed=False, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 use_layerscale=False, 
                 normalize_modulator=False, 
                 use_checkpoint=False,                  
        ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, 
            use_conv_embed=use_conv_embed, is_stem=True)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_window=focal_windows[i_layer], 
                focal_level=focal_levels[i_layer], 
                use_conv_embed=use_conv_embed,
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator, 
                use_layerscale=use_layerscale, 
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = tensor_list.tensors
        tic = time.time()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        # outs = []
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                # outs.append(out)
                outs["res{}".format(i + 2)] = out
        toc = time.time()

        # # collect for nesttensors
        # outs_dict = {}
        # for idx, out_i in enumerate(outs):
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
        #     outs_dict[idx] = NestedTensor(out_i, mask)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()


@BACKBONE_REGISTRY.register()
class D2FocalNet(FocalNet, Backbone):
    def __init__(self, cfg, input_shape):
        kw = cfg.MODEL.FOCAL
        assert kw.modelname in ['focalnet_L_384_22k', 'focalnet_L_384_22k_fl4', 'focalnet_XL_384_22k']
        kw = cfg.MODEL.FOCAL
        if 'focal_levels' in kw:
            kw['focal_levels'] = [kw['focal_levels']] * 4

        if 'focal_windows' in kw:
            kw['focal_windows'] = [kw['focal_windows']] * 4

        model_para_dict = {
            'focalnet_L_384_22k': dict(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                focal_levels=kw.get('focal_levels', [3, 3, 3, 3]),
                focal_windows=kw.get('focal_windows', [5, 5, 5, 5]),
                use_conv_embed=True,
                use_postln=True,
                use_postln_in_modulation=False,
                use_layerscale=True,
                normalize_modulator=False,
            ),
            'focalnet_L_384_22k_fl4': dict(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                focal_levels=kw.get('focal_levels', [4, 4, 4, 4]),
                focal_windows=kw.get('focal_windows', [3, 3, 3, 3]),
                use_conv_embed=True,
                use_postln=True,
                use_postln_in_modulation=False,
                use_layerscale=True,
                normalize_modulator=True,
            ),
            'focalnet_XL_384_22k': dict(
                embed_dim=256,
                depths=[2, 2, 18, 2],
                focal_levels=kw.get('focal_levels', [3, 3, 3, 3]),
                focal_windows=kw.get('focal_windows', [5, 5, 5, 5]),
                use_conv_embed=True,
                use_postln=True,
                use_postln_in_modulation=False,
                use_layerscale=True,
                normalize_modulator=False,
            ),
            'focalnet_huge_224_22k': dict(
                embed_dim=352,
                depths=[2, 2, 18, 2],
                focal_levels=kw.get('focal_levels', [3, 3, 3, 3]),
                focal_windows=kw.get('focal_windows', [5, 5, 5, 5]),
                use_conv_embed=True,
                use_postln=True,
                use_postln_in_modulation=False,
                use_layerscale=True,
                normalize_modulator=False,
            ),
        }

        kw_cgf = model_para_dict[kw.modelname]
        kw1 = {k:v for k, v in kw.items() if 'modelname' not in k and 'out_features' not in k}
        kw_cgf.update(kw1)

        super().__init__(**kw_cgf)


        self._out_features = kw.out_features

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

def build_focalnet(modelname, **kw):
    assert modelname in ['focalnet_L_384_22k', 'focalnet_L_384_22k_fl4', 'focalnet_XL_384_22k']

    if 'focal_levels' in kw:
        kw['focal_levels'] = [kw['focal_levels']] * 4

    if 'focal_windows' in kw:
        kw['focal_windows'] = [kw['focal_windows']] * 4

    model_para_dict = {
        'focalnet_L_384_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]), 
            focal_windows=kw.get('focal_windows', [5, 5, 5, 5]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),
        'focalnet_L_384_22k_fl4': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [4, 4, 4, 4]), 
            focal_windows=kw.get('focal_windows', [3, 3, 3, 3]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=True, 
        ),
        'focalnet_XL_384_22k': dict(
            embed_dim=256,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]), 
            focal_windows=kw.get('focal_windows', [5, 5, 5, 5]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),   
        'focalnet_huge_224_22k': dict(
            embed_dim=352,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]), 
            focal_windows=kw.get('focal_windows', [5, 5, 5, 5]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),                
    }

    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = FocalNet(**kw_cgf)
    return model