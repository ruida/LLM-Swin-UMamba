import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer

from .llava_text_tower import LLaVATextTower

# R2Gen reporting head
from nnunetv2.nets.reporting import R2GenFromSwin, R2GenArgs, Tokenizer


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Mlp(nn.Module):
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


class DropPath_(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath_, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return DropPath(self.drop_prob)(x)


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (H % 2 != 0) or (W % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not possible for SWINTransformer; resize to a integer multiple of 2")
            if H % 2 != 0:
                SHAPE_FIX[0] = H - 1
            if W % 2 != 0:
                SHAPE_FIX[1] = W - 1

            x = x[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention2D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if not isinstance(window_size, int) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # (rest of your attention implementation remains unchanged)


class BasicLayer2D(nn.Module):
    # (unchanged)
    pass


class VSSMEncoder(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16, drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        num_heads_fusion=[1, 1, 1, 1],
        fusion_drop=0.0,
        **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0], norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ape = False

        # (rest unchanged)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = [x]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)
        return x_ret


class SwinUMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        feat_size=[48, 96, 192, 384, 768],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        res_block: bool = True,
        spatial_dims=2,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.deep_supervision = deep_supervision

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )
        self.spatial_dims = spatial_dims
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=feat_size[0])

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder6 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.feat_size[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # segmentation heads (unchanged)
        self.out_layers = nn.ModuleList([
            UnetOutBlock(spatial_dims, self.feat_size[0], out_chans)
        ])
        if deep_supervision:
            self.out_layers = nn.ModuleList([
                UnetOutBlock(spatial_dims, self.feat_size[0], out_chans),
                UnetOutBlock(spatial_dims, self.feat_size[1], out_chans),
                UnetOutBlock(spatial_dims, self.feat_size[2], out_chans),
                UnetOutBlock(spatial_dims, self.feat_size[3], out_chans),
            ])

        # --- your existing LLaVA text tower (unchanged) ---
        self.text_encoder = LLaVATextTower(
            model_name="meta-llama/Llama-3-8b-Instruct",
            max_length=256,
            normalize=True,
        )
        txt_dim = getattr(self.text_encoder, "output_dim",
                          getattr(self.text_encoder, "hidden_size", 4096))
        self.lang_proj = nn.ModuleList([
            nn.Linear(txt_dim, self.feat_size[0]),
            nn.Linear(txt_dim, self.feat_size[1]),
            nn.Linear(txt_dim, self.feat_size[2]),
            nn.Linear(txt_dim, self.feat_size[3]),
            nn.Linear(txt_dim, self.feat_size[4]),
        ])

        # === R2Gen reporting head (lazy init; segmentation/text encoding unchanged) ===
        self.enable_r2gen_generation: bool = True
        self._last_deep_feature = None  # cached deepest feature for report head

        # Non-invasive cache via encoder forward hook (no changes to forward())
        def _cache_deep_feature(module, inp, out):
            try:
                self._last_deep_feature = out[-1] if isinstance(out, (list, tuple)) else out
            except Exception:
                self._last_deep_feature = None
        try:
            self.vssm_encoder.register_forward_hook(_cache_deep_feature)
        except Exception:
            pass

        # Initialize later with a proper R2Gen tokenizer via `init_r2gen(...)`
        self._report_head = None

    def forward(self, x_in, report_text):
        """
        output = self.network(data, report_text)
        x_in: (B, C, H, W)
        report_text: list[str] or str
        """
        device = x_in.device
        B = x_in.shape[0]
        if not isinstance(report_text, str):
            raise TypeError('report_text must be a single string')

        # Text encoding for segmentation fusion (LLaVA tower)
        if hasattr(self, "text_encoder") and (self.text_encoder is not None):
            lang_feat = self.text_encoder(report_text)   # (B, txt_dim)
        else:
            in_dim = getattr(self.lang_proj[0], "in_features")
            lang_feat = x_in.new_zeros((B, in_dim))

        # Vision encoder
        x = self.stem(x_in) if hasattr(self, "stem") else x_in
        vss_outs = self.vssm_encoder(x)

        # Auxiliary encoders (kept from your pipeline)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(vss_outs[0])
        enc3 = self.encoder3(vss_outs[1])
        enc4 = self.encoder4(vss_outs[2])
        enc5 = self.encoder5(vss_outs[3])

        enc_hidden = vss_outs[-1]  # deepest (also cached by the hook)

        # Decoder with language fusion (unchanged)
        def _broadcast_add(feat, proj_idx):
            add = self.lang_proj[proj_idx](lang_feat)
            return feat + add.unsqueeze(-1).unsqueeze(-1).expand_as(feat)

        dec4 = self.decoder6(enc_hidden, enc5)
        dec4 = _broadcast_add(dec4, 4)

        dec3 = self.decoder5(dec4, enc4)
        dec3 = _broadcast_add(dec3, 3)

        dec2 = self.decoder4(dec3, enc3)
        dec2 = _broadcast_add(dec2, 2)

        dec1 = self.decoder3(dec2, enc2)
        dec1 = _broadcast_add(dec1, 1)

        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)
        dec_out = _broadcast_add(dec_out, 0)

        # Segmentation logits (unchanged)
        if getattr(self, "deep_supervision", False):
            return [self.out_layers[i](feat) for i, feat in enumerate([dec_out, dec1, dec2, dec3])]
        else:
            return self.out_layers[0](dec_out)

    def init_r2gen(self, tokenizer, d_vf: int = 2048, token_hw: Tuple[int, int] = (7, 7), args: Optional[R2GenArgs] = None):
        """Attach the R2Gen-Mamba head *without* touching segmentation/text encoder.
        Args:
            tokenizer: R2Gen tokenizer instance (from nnunetv2.nets.reporting.r2gen_mamba_orig.tokenizers.Tokenizer).
            d_vf: visual feature dim expected by R2Gen
            token_hw: pooled token grid size, e.g., (7,7)
            args: R2GenArgs or None for defaults
        """
        self._report_head = R2GenFromSwin(
            tokenizer=tokenizer,
            in_channels=self.feat_size[-1],
            d_vf=d_vf,
            token_hw=token_hw,
            args=args if args is not None else R2GenArgs(d_vf=d_vf)
        )
        return self._report_head

    def forward_report_train(self, targets):
        """
        Train the R2Gen head strictly from the cached deepest visual features.

        NOTE:
          - Segmentation & text encoding paths remain **unchanged**.
          - `report_text` is **not** used here.
          - `targets` must be either:
              (a) LongTensor of token IDs shaped [B, T] with BOS/EOS/PAD per R2Gen, or
              (b) list[str] reports; in this case, a tokenizer **must** have been provided when initializing the R2Gen head.
        """
        if not (getattr(self, "enable_r2gen_generation", False) and (self._report_head is not None)):
            raise RuntimeError("R2Gen report head is disabled or not initialized. Call `init_r2gen(tokenizer, ...)` first.")
        if self._last_deep_feature is None:
            raise RuntimeError("Run forward(images, report_text) first to populate cached visual features.")

        # If raw strings are provided, rely on the R2Gen head to encode (requires tokenizer)
        return self._report_head(self._last_deep_feature, targets=targets, mode="train")

    @torch.no_grad()
    def forward_report_infer(self, gen_kwargs: Optional[Dict] = None):
        """
        Generate report text from the last cached visual feature.
        gen_kwargs: e.g., {"sample_method":"beam_search","beam_size":3,"max_seq_length":64,"temperature":1.0}
        """
        if not (getattr(self, "enable_r2gen_generation", False) and (self._report_head is not None)):
            raise RuntimeError("R2Gen report head is disabled or not initialized. Call `init_r2gen(tokenizer, ...)` first.")
        if self._last_deep_feature is None:
            raise RuntimeError("Call forward(images, report_text) before forward_report_infer().")

        return self._report_head(self._last_deep_feature, mode="sample", **(gen_kwargs or {}))

    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True

    @staticmethod
    def load_pretrained_ckpt(
        model,
        ckpt_path="./data/pretrained/vmamba/vmamba_tiny_e292.pth"
    ):
        print(f"Loading weights from: {ckpt_path}")
        skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias",
                       "patch_embed.proj.weight", "patch_embed.proj.bias",
                       "patch_embed.norm.weight", "patch_embed.norm.weight"]

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_dict = model.state_dict()
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            if k in model_dict and (k not in skip_params):
                state_dict[k] = v
            else:
                print("skip:", k)
        model.load_state_dict(state_dict, strict=False)
        return model


def create_model(plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_input_channels: Union[int, List[int]], deep_supervision: bool = False, use_pretrain: bool = False):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = SwinUMamba(
        in_chans=num_input_channels,
        out_chans=label_manager.num_segmentation_heads,
        feat_size=[48, 96, 192, 384, 768],
        deep_supervision=deep_supervision,
        hidden_size=768,
    )

    if use_pretrain:
        model = SwinUMamba.load_pretrained_ckpt(model)

    return model
