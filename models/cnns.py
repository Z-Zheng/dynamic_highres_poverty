# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def non_mse_loss(pr, gt):
    diff = torch.squeeze(pr) - gt
    not_nan = ~torch.isnan(diff)
    losses = diff.masked_select(not_nan) ** 2
    return losses.mean()


def r_square(pr, gt):
    diff = torch.squeeze(pr) - gt
    not_nan = ~torch.isnan(diff)
    rss = (diff.masked_select(not_nan) ** 2).sum()

    gt = gt.masked_select(not_nan)
    tss = ((gt - torch.mean(gt)) ** 2).sum()
    r2 = torch.maximum(torch.zeros_like(rss), 1. - rss / tss)

    return r2


@er.registry.MODEL.register()
class SiameseEncoderOnly(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.encoder_name.startswith('swin'):
            from module.tv_swin import TVSwinTransformer
            import torchvision.models as tvm
            weights = {
                'swin_t': tvm.Swin_T_Weights.IMAGENET1K_V1,
                'swin_b': tvm.Swin_B_Weights.IMAGENET1K_V1,
                'swin_v2_t': tvm.Swin_V2_T_Weights.IMAGENET1K_V1,
                'swin_v2_b': tvm.Swin_V2_B_Weights.IMAGENET1K_V1,
            }
            model = TVSwinTransformer(dict(
                name=self.cfg.encoder_name,
                weights=weights[self.cfg.encoder_name]
            ))
            patch_first_conv(model.swin.features[0], self.cfg.in_channels, 3, True)
            self.encoder = model
            channels = {
                'swin_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
                'swin_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
                'swin_v2_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
                'swin_v2_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
            }
            last_channels = channels[self.cfg.encoder_name][-1]
        elif self.cfg.encoder_name.startswith('torchgeo_r50_landsat'):
            import torchgeo.models
            from torchgeo.models.resnet import ResNet50_Weights
            from types import MethodType
            self.encoder = torchgeo.models.resnet50(weights=ResNet50_Weights.LANDSAT_ETM_SR_MOCO)
            last_channels = 2048

            def _forward(self, x):
                x = self.forward_features(x)
                return [x]

            self.encoder.forward = MethodType(_forward, self.encoder)
        else:
            self.encoder = smp.encoders.get_encoder(
                name=self.cfg.encoder_name,
                in_channels=self.cfg.in_channels,
                weights=self.cfg.encoder_weights
            )
            last_channels = self.encoder.out_channels[-1]

        self.mlp = nn.Sequential(
            nn.Linear(2 * last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
            nn.GELU(),
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
        )
        self.linear = nn.Linear(last_channels, 1, bias=True)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        feature = self.encoder(x)[-1]
        feature = torch.mean(feature, dim=(2, 3))
        awi = self.linear(self.mlp(rearrange(feature, '(b t) c -> b (t c)', t=2)))

        if self.training:
            gt_awi1, gt_awi2 = y['awi']
            gt_awi = gt_awi2 - gt_awi1
            return {
                'mse_loss': non_mse_loss(awi.reshape(-1), gt_awi.reshape(-1)),
                'train/r2': r_square(awi.reshape(-1), gt_awi.reshape(-1)),
            }
        return awi

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet18',
            in_channels=6,
            encoder_weights='imagenet',
        ))


@er.registry.MODEL.register()
class EncoderOnly(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.encoder_name.startswith('swin'):
            from models.tv_swin import TVSwinTransformer
            import torchvision.models as tvm
            weights = {
                'swin_t': tvm.Swin_T_Weights.IMAGENET1K_V1,
                'swin_b': tvm.Swin_B_Weights.IMAGENET1K_V1,
                'swin_v2_t': tvm.Swin_V2_T_Weights.IMAGENET1K_V1,
                'swin_v2_b': tvm.Swin_V2_B_Weights.IMAGENET1K_V1,
            }
            model = TVSwinTransformer(dict(
                name=self.cfg.encoder_name,
                weights=weights[self.cfg.encoder_name]
            ))
            patch_first_conv(model.swin.features[0], self.cfg.in_channels, 3, True)
            self.encoder = model
            channels = {
                'swin_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
                'swin_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
                'swin_v2_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
                'swin_v2_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
            }
            last_channels = channels[self.cfg.encoder_name][-1]
        elif self.cfg.encoder_name.startswith('torchgeo_r50_landsat'):
            import torchgeo.models
            from torchgeo.models.resnet import ResNet50_Weights
            from types import MethodType
            self.encoder = torchgeo.models.resnet50(weights=ResNet50_Weights.LANDSAT_ETM_SR_MOCO)
            last_channels = 2048

            def _forward(self, x):
                x = self.forward_features(x)
                return [x]

            self.encoder.forward = MethodType(_forward, self.encoder)
        else:
            self.encoder = smp.encoders.get_encoder(
                name=self.cfg.encoder_name,
                in_channels=self.cfg.in_channels,
                weights=self.cfg.encoder_weights
            )
            last_channels = self.encoder.out_channels[-1]

        self.mlp = nn.Sequential(
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
            nn.GELU(),
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
        )
        self.linear = nn.Linear(last_channels, 1, bias=True)
        self.init_from_weight_file()

    def forward(self, x, y=None):
        feature = self.encoder(x)[-1]
        feature = torch.mean(feature, dim=(2, 3))
        awi = self.linear(self.mlp(feature))

        if self.training:
            gt_awi = y['awi']
            return {
                'mse_loss': non_mse_loss(awi.reshape(-1), gt_awi.reshape(-1)),
                'train/r2': r_square(awi.reshape(-1), gt_awi.reshape(-1)),
            }
        return awi

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet18',
            in_channels=6,
            encoder_weights='imagenet',
        ))
