# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from .head_modules import UnetExtractor, AppearanceTransformer, _init_weights
from .postprocess import postprocess

    # def __init__(self,
    #              num_channels: int = 1,
    #              stride_level: int = 1,
    #              patch_size: Union[int, Tuple[int, int]] = 16,
    #              main_tasks: Iterable[str] = ('rgb',),
    #              hooks: List[int] = [2, 5, 8, 11],
    #              layer_dims: List[int] = [96, 192, 384, 768],
    #              feature_dim: int = 256,
    #              last_dim: int = 32,
    #              use_bn: bool = False,
    #              dim_tokens_enc: Optional[int] = None,
    #              head_type: str = 'regression',
    #              output_width_ratio=1,

class VGGT_DPT_GS_Head(DPTHead):
    def __init__(self, 
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation, features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio)
        
        head_features_1 = 128
        head_features_2 = 128 if output_dim > 50 else 32 # sh=0, head_features_2 = 32; sh=4, head_features_2 = 128
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_2, 7, 1, 3),
            nn.ReLU(),
        )
        
        self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )
        
    def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs, patch_start_idx: int = 5, image_size=None, conf=None, frames_chunk_size: int = 8, exposure=None):
        # H, W = input_info['image_size']
        B, S, _, H, W = imgs.shape
        image_size = self.image_size if image_size is None else image_size
    
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, patch_start_idx, exposure=exposure)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            chunk_output = self._forward_impl(
                encoder_tokens, imgs, patch_start_idx, frames_start_idx, frames_end_idx, exposure=exposure
            )
            all_preds.append(chunk_output)
        
        # Concatenate results along the sequence dimension
        if exposure==None:
            return torch.cat(all_preds, dim=1)
        return torch.cat(all_preds, dim=0)
    
    def _forward_impl(self, encoder_tokens: List[torch.Tensor], imgs, patch_start_idx: int = 5, frames_start_idx: int = None, frames_end_idx: int = None, exposure=None):
        
        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape

        patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()
            
            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        direct_img_feat = self.input_merger(imgs.flatten(0,1))
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        if exposure == None:
            out = out + direct_img_feat

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if exposure == None:
            out = self.scratch.output_conv2(out)
            out = out.view(B, S, *out.shape[1:])
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExposurePositionalEncoding(nn.Module):
    """正弦位置编码，用于将 scalar 曝光值映射为向量"""
    def __init__(self, d_model=128, max_stop=8.0):
        super().__init__()
        self.max_stop = max_stop
        div_term = torch.exp(math.log(2.0) * (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """x: [B, ...], values in stops (log2 space)"""
        x_norm = x.unsqueeze(-1) * (math.pi / self.max_stop)
        pe = torch.zeros(*x.shape, self.div_term.shape[0] * 2, device=x.device)
        pe[..., 0::2] = torch.sin(x_norm * self.div_term)
        pe[..., 1::2] = torch.cos(x_norm * self.div_term)
        return pe
    
class GuidedFilterUpsampling(nn.Module):
    """利用高分辨率 RGB 图作为引导，上采样低分 HDR 特征"""
    def __init__(self, in_channels=128, out_channels=128): # 假设 patch_size=16 (224/14)
        super().__init__()
        self.proj_guide = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 3),
            nn.ReLU()
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, 1, 3),
            nn.ReLU()
        )

    def forward(self, lr_feat, guide):
        # lr_feat: [BV, C, h, w], guide: [BV, C, H, W]
        H, W = guide.shape[-2:]
        h, w = lr_feat.shape[-2:]
        x_up = F.interpolate(lr_feat, size=(H, W), mode="bilinear", align_corners=False)
        g_low = F.interpolate(guide, size=(h//2,w//2), mode="bilinear", align_corners=False)
        g_low = F.interpolate(g_low, size=(H, W), mode="bilinear", align_corners=False)
        g_high = self.proj_guide(guide - g_low)
        return self.proj_out(x_up + g_high)
        # return self.proj_out(x_up)

class ExposureAwareGeoAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.v_mean_proj = nn.Linear(dim, dim)
        self.v_var_proj = nn.Linear(dim, dim)
        self.mean_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.var_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.exp_film = nn.Sequential(
            nn.Linear(3*dim, dim), nn.ReLU(),
            nn.Linear(dim, dim * 2)  # gamma, beta
        )
        # self.adain_pred = nn.Sequential(
        #     nn.Linear(2*dim, dim), nn.ReLU(),
        #     nn.Linear(dim, dim * 2)  # gamma, beta
        # )

    def forward(self, x, exp_flat, attn, V, logt):
        B, N, C = x.shape
        M = N // V
        
        x_norm = self.ln(x)
        x_per_view = x_norm.view(B, V, M, C)
        v_mean = self.v_mean_proj(x_norm)

        gamma, beta = self.exp_film(
            torch.cat([exp_flat.detach(),
                       x_per_view.mean(dim=2).detach(), 
                       x_per_view.mean(dim=2).mean(dim=1, keepdim=True).expand(-1, V, -1).detach()
                       ], dim=-1)  # [B, V, 3C]
        ).chunk(2, dim=-1)  # [B, V, C]
        gamma = gamma.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N, C)
        beta = beta.unsqueeze(2).expand(-1, -1, M, -1).reshape(B, N, C)
        v_mean = v_mean * (1 + gamma) + beta
        
        v_var = self.v_var_proj(x_norm)
        attn_grouped = attn.view(B, N, V, M)  # [B, N, V, M]
        
        # ---- 可见性权重 ----
        vis_w = attn_grouped.sum(dim=-1)  # [B, N, V]
        vis_w = vis_w / (vis_w.sum(dim=-1, keepdim=True) + 1e-6)
        
        # ---- mean路径: 吸收权重，单次 matmul ----
        attn_weighted = attn_grouped * vis_w.unsqueeze(-1)  # [B, N, V, M]
        attn_weighted = attn_weighted.reshape(B, N, V * M)  # [B, N, N]
        weighted_mean = (attn_weighted @ v_mean.bfloat16()).to(x.dtype)  # [B, N, C]
        
        # ---- var路径: bmm 替代 einsum ----
        attn_t = attn_grouped.permute(0, 2, 1, 3).reshape(B * V, N, M)  # [BV, N, M]
        vv = v_var.view(B, V, M, C).reshape(B * V, M, C)                 # [BV, M, C]
        agg_var = torch.bmm(attn_t, vv.bfloat16()).to(x.dtype)           # [BV, N, C]
        agg_var = agg_var.view(B, V, N, C).permute(0, 2, 1, 3)           # [B, N, V, C]
        
        vis_w_exp = vis_w.unsqueeze(-1)                                   # [B, N, V, 1]
        var_center = (agg_var * vis_w_exp).sum(dim=2)                     # [B, N, C]
        diff = agg_var - var_center.unsqueeze(2)
        weighted_var = (diff.pow(2) * vis_w_exp).sum(dim=2)               # [B, N, C]
        
        # ---- 曝光尺度归一化 ----
        exp_range = (logt.max(dim=1).values - logt.min(dim=1).values)
        exp_range = exp_range.clamp(min=0.5).view(B, 1, 1)
        weighted_var = weighted_var / exp_range
        
        scene_feat = self.mean_out(weighted_mean)
        # scene_feat = self.mean_out(v_mean)
        sensitivity = self.var_out(weighted_var.to(weighted_mean.dtype))
        
        return x + scene_feat, sensitivity

from ..vggt.layers.patch_embed import PatchEmbed
import copy
class HDR_Head(nn.Module):
    def __init__(self, rgb_conv_encoder, rad_dim=128, reg_num=5, patch_size=14):
        super().__init__()
        self.reg_num = reg_num
        self.dim = rad_dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(3, rad_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = ExposurePositionalEncoding(d_model=rad_dim, max_stop=8.0)
        self.regression = ExposureAwareGeoAttention(dim=rad_dim)

        self.guide_encoder = copy.deepcopy(rgb_conv_encoder)
        self.scene_conv = nn.Sequential(nn.Conv2d(rad_dim, rad_dim, 3, 1, 1), nn.ReLU())
        self.sensitivity_conv = nn.Sequential(nn.Conv2d(rad_dim, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0))
        self.hdr_upsample = GuidedFilterUpsampling()
        # self.sen_upsample = GuidedFilterUpsampling(out_channels=4)

    def forward(self, imgs, exposure_list, geo_q, geo_k, scale=64**-0.5):
        B, V, _, H, W = imgs.shape # [B,V,3,448,448]
        h, w = H // self.patch_size, W // self.patch_size

        # --- 1. Exposure Encoding ---
        t = torch.stack(exposure_list, dim=1).to(imgs.device) 
        log_t = torch.log2(t.clamp_min(1e-8)) 
        mid_logt = (log_t.min(1, keepdim=True).values + log_t.max(1, keepdim=True).values) / 2
        rel_logt = log_t - mid_logt
        rel_logt = self.pos_embed(rel_logt)

        # --- 2. Patch Embed (Remove FiLM) ---
        x = self.patch_embed(imgs.flatten(0, 1)) # [BV, C, h, w]
        x = rearrange(x, '(b v) c h w -> b (v h w) c', b=B, v=V)
        exp_emb = rel_logt
        # exp_emb = rel_logt.unsqueeze(2).expand(-1, -1, h*w, -1)
        # exp_emb = rearrange(exp_emb, 'b v hw c -> b (v hw) c')
        
        # --- 3. Attention & Regression ---
        with torch.no_grad():
            attn = (geo_q.bfloat16() * scale) @ geo_k.transpose(-2, -1).bfloat16()
            attn = attn.softmax(dim=-1).mean(dim=1) # [B,N,N]
            attn = rearrange(attn, 'b (v1 n1) (v2 n2) -> b v1 n1 v2 n2', v1=V, v2=V)
            attn = attn[:, :, self.reg_num:, :, self.reg_num:]
            attn = rearrange(attn, 'b v1 n1 v2 n2 -> b (v1 n1) (v2 n2)').contiguous()
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        scene_feat, sensitivity = self.regression(x, exp_emb, attn, V, log_t)
        del attn

        # 场景特征 → 3D 重建
        hr_guide = self.guide_encoder(imgs.flatten(0, 1))
        scene_feat = rearrange(scene_feat, 'b (v h w) c -> (b v) c h w', v=V, h=h, w=w)
        scene_feat = self.hdr_upsample(self.scene_conv(scene_feat), hr_guide)

        # 曝光敏感度 → Gaussian local feature → tonemapper
        sensitivity = rearrange(sensitivity, 'b (v h w) c -> (b v) c h w', v=V, h=h, w=w)
        # sensitivity = self.sen_upsample(self.sensitivity_conv(sensitivity), hr_guide)
        sensitivity = F.interpolate(self.sensitivity_conv(sensitivity), size=(H, W), mode='bilinear', align_corners=False)
        
        return scene_feat, sensitivity, mid_logt, hr_guide, rel_logt

class HyperNetwork(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=128, tm_hidden=64):
        super().__init__()
        self.tm_param_dim = 30 * tm_hidden + 6
        self.exp_proj = nn.Linear(feat_dim, feat_dim)
        self.enc = nn.Sequential(
            nn.Conv2d(feat_dim + 88 + feat_dim, hidden_dim, 14, 14, 0), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.tm_param_dim)
        )

    def forward(self, ldr_feats, exp_emb, hdr_feats):
        """
        ldr_feats: [B, N, 128, H, W]
        exp_emb:   [B, N, 128]
        hdr_feats: [B, N, 88, H, W]
        """
        B, N, C, H, W = ldr_feats.shape
        
        exp_spatial = self.exp_proj(exp_emb)
        exp_spatial = exp_spatial.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        
        x = torch.cat([
            ldr_feats.flatten(0, 1),
            hdr_feats.flatten(0, 1),
            exp_spatial.flatten(0, 1)
        ], dim=1)
        
        x = self.enc(x)  # [BN, hidden, h, w]

        # 空间 / View avg+max
        x_avg = x.mean(dim=[-2, -1])              # [BN, hidden]
        x_max = x.amax(dim=[-2, -1])              # [BN, hidden]
        x = torch.cat([x_avg, x_max], dim=-1)     # [BN, hidden*2]
        x = x.view(B, N, -1)                      # [B, N, hidden*2]
        x = torch.cat([
            x.mean(dim=1),                        # [B, hidden*2]
            x.max(dim=1).values                   # [B, hidden*2]
        ], dim=-1)                                 # [B, hidden*4]
        return self.out(x)
    
class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        
        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, depths, imgs, img_info, conf=None):
        out, interm_feats = self.dpt(x, depths, imgs, image_size=(img_info[0], img_info[1]), conf=conf)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out, interm_feats

def create_gs_dpt_head(net, has_conf=False, out_nchan=3, postprocess_func=postprocess):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = net.feature_dim
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    try:    
        patch_size = net.patch_size
    except:
        patch_size = (16, 16)

    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                patch_size=patch_size,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='gs_params')