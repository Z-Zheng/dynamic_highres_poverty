# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from models.tv_swin import TVSwinTransformer
import torchvision.models as tvm
import ever as er
import ever.module as M
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.ops import StochasticDepth, MLP
from timm.models.vision_transformer import vit_base_patch8_224


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
class TransformerMLP(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
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

        # awi + change
        self.awi_linear = nn.Linear(last_channels, 1, bias=True)
        self.change_linear = nn.Linear(last_channels, 1, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
            nn.GELU(),
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
        )

        self.temporal_mlp = nn.Sequential(
            nn.Linear(2 * last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
            nn.GELU(),
            nn.Linear(last_channels, last_channels, bias=False),
            nn.LayerNorm(last_channels),
        )

    def forward(self, x, y=None):
        inc = x.size(1)
        if inc // self.cfg.in_channels == 1:
            feature = self.encoder(x)[-1]
            feature = torch.mean(feature, dim=(2, 3))
            feature = self.mlp(feature)
            c_awi = self.awi_linear(feature).squeeze()
        elif inc // self.cfg.in_channels == 2:
            x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
            feature = self.encoder(x)[-1]
            feature = torch.mean(feature, dim=(2, 3))
            feature = self.mlp(feature)
            feature1, feature2 = rearrange(feature, '(b t) c -> t b c', t=2)

            c_feature = rearrange(feature, '(b t) c -> b (t c)', t=2)
            c_feature = self.temporal_mlp(c_feature)

            # t1_awi = self.awi_linear(feature1).squeeze()
            # t2_awi = self.awi_linear(feature2).squeeze()
            c_awi = self.change_linear(c_feature).squeeze()
            # c_awi = 0.5 * c_awi + 0.5 * (t2_awi - t1_awi)

        if self.training:
            if inc // self.cfg.in_channels == 1:
                gt_awi = y['awi']
                return {
                    'mse_loss': non_mse_loss(c_awi.reshape(-1), gt_awi.reshape(-1)),
                    'train/r2': r_square(c_awi.reshape(-1), gt_awi.reshape(-1)),
                }
            elif inc // self.cfg.in_channels == 2:
                gt_c = y['awi']

                gt_t1 = y['t1_awi']
                gt_t2 = y['t2_awi']
                return {
                    'train/c_mse_loss': non_mse_loss(c_awi.reshape(-1), gt_c.reshape(-1)),
                    # 'train/t1_mse_loss': non_mse_loss(t1_awi.reshape(-1), gt_t1.reshape(-1)),
                    # 'train/t2_mse_loss': non_mse_loss(t2_awi.reshape(-1), gt_t2.reshape(-1)),
                    'train/r2': r_square(c_awi.reshape(-1), gt_c.reshape(-1)),
                }
            else:
                raise NotImplementedError

        return c_awi

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='swin_v2_t',
            in_channels=6,
            encoder_weights='imagenet',
        ))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = rearrange(q, 'b n (h ch) -> b h n ch', h=self.num_heads)
        k = rearrange(k, 'b n (h ch) -> b h n ch', h=self.num_heads)
        v = rearrange(v, 'b n (h ch) -> b h n ch', h=self.num_heads)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        out = rearrange(out, 'b h n ch -> b n (h ch)')
        out = self.out_proj(out)
        return out


class Conditioning(nn.Module):
    def __init__(self, dim, x_dim, g_dim, num_heads, drop_path_rate=0.):
        super().__init__()
        self.dim = dim
        self.x_dim = x_dim
        self.cross_attn_feature_to_token = Attention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = StochasticDepth(drop_path_rate, mode='row')
        self.mlp = MLP(dim, [int(dim * 4), dim], activation_layer=nn.GELU, inplace=None, dropout=0.)

        self.g_encoder = nn.Sequential(
            nn.Linear(g_dim, dim, bias=True),
            nn.LayerNorm(dim),
        )
        if dim != x_dim:
            self.linear = nn.Linear(x_dim, dim)

    def forward(self, q, x, g):
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.mean(x, dim=1, keepdim=True)
        g = self.g_encoder(g)
        if self.dim != self.x_dim:
            x = self.linear(x)
        features = torch.cat([x, g], dim=1)

        q = q + self.drop_path(self.norm1(self.cross_attn_feature_to_token(q, features, features)))
        q = q + self.drop_path(self.norm2(self.mlp(q)))
        return q


@er.registry.MODEL.register()
class MultimodalTransformerMLP(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
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
        # self.encoder.requires_grad_(False)
        channels = {
            'swin_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
            'swin_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
            'swin_v2_t': [3, 0] + [96 * (2 ** i) for i in range(4)],
            'swin_v2_b': [3, 0] + [128 * (2 ** i) for i in range(4)],
        }
        last_channels = channels[self.cfg.encoder_name][-1]

        hs = [3, 6, 12, 24]
        self.ada_blocks = nn.ModuleList(
            [Conditioning(last_channels, x_dim=c, g_dim=128, num_heads=nh, drop_path_rate=0.3) for nh, c in
             zip(hs, channels[self.cfg.encoder_name][2:])]
        )
        self.g_embedding = nn.Sequential(
            nn.Linear(self.cfg.g_channels, 64, bias=True),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128, bias=True),
            nn.LayerNorm(128),
        )
        self.token = nn.Embedding(1, last_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(last_channels),
            nn.Dropout(p=0.3),
            nn.Linear(last_channels, 1)
        )

    def forward(self, x, g, y=None):
        g_emb = self.g_embedding(g)
        inc = x.size(1)
        token_embed = self.token.weight.unsqueeze(0).expand(x.size(0), -1, -1)

        if inc // self.cfg.in_channels == 1:
            for i, block in enumerate(self.encoder.swin.get_stages()):
                x = block(x)
                x_feat = x.permute(0, 3, 1, 2).contiguous()
                token_embed = self.ada_blocks[i](q=token_embed, x=x_feat, g=g_emb)
            c_awi = self.head(token_embed)
        else:
            raise NotImplementedError

        if self.training:
            if inc // self.cfg.in_channels == 1:
                gt_awi = y['awi']
                return {
                    'mse_loss': non_mse_loss(c_awi.reshape(-1), gt_awi.reshape(-1)),
                    'train/r2': r_square(c_awi.reshape(-1), gt_awi.reshape(-1)),
                }
            else:
                raise NotImplementedError
        return c_awi

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='swin_v2_t',
            in_channels=6,
            g_channels=15,
            encoder_weights='imagenet',
        ))
