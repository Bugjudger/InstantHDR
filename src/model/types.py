from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

# def tonemapping(log_rad, params, hidden=64):
#     # log_rad: [B, V, 3, H, W]
#     # params : [B, 9*hidden+3]
#     B, V, C, H, W = log_rad.shape
#     assert C == 3
#     assert params.shape == (B, 9*hidden + 3)

#     # 拆参数
#     p = params.view(B, 3, 3*hidden + 1)   # [B,3,(W1+b1+W2+b2)]
#     W1 = p[:, :, 0:hidden]                # [B,3,h]
#     b1 = p[:, :, hidden:2*hidden]         # [B,3,h]
#     W2 = p[:, :, 2*hidden:3*hidden]       # [B,3,h]
#     b2 = p[:, :, 3*hidden:]               # [B,3,1]

#     # reshape: [B,V,3,H,W] -> [B,3,VHW] -> [B,N,3,1]
#     N = V * H * W
#     log_rad = log_rad.permute(0, 2, 1, 3, 4).contiguous().view(B, 3, N)   # [B,3,N]
#     log_rad = log_rad.permute(0, 2, 1).unsqueeze(-1)                            # [B,N,3,1]

#     # 两层 MLP（逐通道，batch）
#     log_rad = F.relu(log_rad * W1[:, None] + b1[:, None])                       # [B,N,3,h]
#     log_rad = (log_rad * W2[:, None]).sum(-1, keepdim=True) + b2[:, None]       # [B,N,3,1]
#     rgb = torch.sigmoid(log_rad).squeeze(-1)                              # [B,N,3]

#     # # --- 分块处理逻辑 ---
#     # chunk_size = 1000000  # 根据显存大小调整这个值 (比如 10^6)
#     # log_rad_chunks = torch.split(log_rad, chunk_size, dim=1)
#     # rgb_chunks = []
#     # for chunk in log_rad_chunks:
#     #     hidden = F.relu(chunk * W1[:, None] + b1[:, None])          # [B, chunk_N, 3, h]
#     #     out = (hidden * W2[:, None]).sum(-1, keepdim=True) + b2[:, None] # [B, chunk_N, 3, 1]
#     #     rgb_chunks.append(torch.sigmoid(out))
#     # rgb = torch.cat(rgb_chunks, dim=1).squeeze(-1)
        
#     # 还原回 [B,V,3,H,W]
#     rgb = rgb.view(B, V, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()                    # [B,V,3,H,W]
#     # rgb = torch.pow(rgb, 1/2.2)
#     return rgb

def tonemapping(log_rad, local_feat, params, hidden=64, is_training=True):
    B, V, C, H, W = log_rad.shape
    F_dim = local_feat.shape[2] # 4
    params = params.to(log_rad)

    # 1. 展平空间维度，准备输入
    x = log_rad.permute(0, 1, 3, 4, 2).reshape(B, -1, C).unsqueeze(-1)          # [B, N, 3, 1]
    f = local_feat.permute(0, 1, 3, 4, 2).reshape(B, -1, F_dim)                 # [B, N, 4]
    
    # 2. 拆分 Global 和 Local 参数
    dim_g = C * (3 * hidden + 1)
    p_g, p_l = torch.split(params, [dim_g, params.shape[1] - dim_g], dim=1)

    # --- Global Tone Mapping (1 -> h -> 1) ---
    w1, b1, w2, b2 = p_g.view(B, 1, C, -1).split([hidden, hidden, hidden, 1], dim=-1)
    h = F.relu(x * w1 + b1)                                                     # [B, N, 3, h]
    y_glo = torch.sigmoid((h * w2).sum(-1, keepdim=True) + b2)                  # [B, N, 3, 1]

    return y_glo.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
    # --- Local Tone Mapping (5 -> h -> 1) ---
    # 拼接输入: RGB(1) + Feat(4) = 5
    f_exp = f.unsqueeze(2).expand(-1, -1, C, -1)                                # [B, N, 3, 4]
    x_loc = torch.cat([x, f_exp], dim=-1)                           # [B, N, 3, 5]

    w1_dim = (1 + F_dim) * hidden
    w1, b1, w2, b2 = p_l.view(B, 1, C, -1).split([w1_dim, hidden, hidden, 1], dim=-1)
    w1 = w1.view(B, 1, C, 1 + F_dim, hidden)                                    # [B, 1, 3, 5, h]
    h = F.relu(torch.matmul(x_loc.unsqueeze(-2), w1).squeeze(-2) + b1)          # [B, N, 3, h]
    y_loc = torch.sigmoid((h * w2).sum(-1, keepdim=True) + b2)                  # [B, N, 3, 1]

    # --- 融合与还原 ---
    out = (y_glo + (y_loc - 0.5)).clamp(0, 1).squeeze(-1)                         # [B, N, 3]
    if is_training:
        if torch.rand(1).item() < 0.8:
            return y_glo.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
    return out.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()          # [B, V, 3, H, W]

# def tonemapping(log_rad, local_feat, params, scene_params, hidden=64, is_training=True):
#     B, V, C, H, W = log_rad.shape
#     F_dim = local_feat.shape[2] # 4
#     params = params.to(log_rad)
#     g1, b1_film, g2, b2_film = scene_params.chunk(4, dim=-1)

#     x = log_rad.permute(0, 1, 3, 4, 2).reshape(B, -1, C).unsqueeze(-1)          # [B, N, 3, 1]
#     f = local_feat.permute(0, 1, 3, 4, 2).reshape(B, -1, F_dim)                 # [B, N, 4]

#     # FiLM 参数扩维，广播到 [B, 1, 1, hidden]
#     def expand_film(t):
#         return t.view(B, 1, 1, -1)                                           # [B, 1, 1, hidden]

#     dim_g = C * (3 * hidden + 1)
#     p_g, p_l = torch.split(params, [dim_g, params.shape[1] - dim_g], dim=1)

#     # --- Global Tone Mapping ---
#     w1, b1, w2, b2 = p_g.view(B, 1, C, -1).split([hidden, hidden, hidden, 1], dim=-1)
#     h = F.relu(x * w1 + b1)                                                  # [B, N, 3, hidden]
#     h = h * expand_film(g1) + expand_film(b1_film)                           # FiLM ← 加在这里
#     y_glo = torch.sigmoid((h * w2).sum(-1, keepdim=True) + b2)               # [B, N, 3, 1]

#     # --- Local Tone Mapping ---
#     f_exp = f.unsqueeze(2).expand(-1, -1, C, -1)
#     x_loc = torch.cat([x, f_exp], dim=-1)                                    # [B, N, 3, 5]
#     w1_dim = (1 + F_dim) * hidden
#     w1, b1, w2, b2 = p_l.view(B, 1, C, -1).split([w1_dim, hidden, hidden, 1], dim=-1)
#     w1 = w1.view(B, 1, C, 1 + F_dim, hidden)
#     h = F.relu(torch.matmul(x_loc.unsqueeze(-2), w1).squeeze(-2) + b1)      # [B, N, 3, hidden]
#     h = h * expand_film(g2) + expand_film(b2_film)                           # FiLM ← 加在这里
#     y_loc = torch.sigmoid((h * w2).sum(-1, keepdim=True) + b2)               # [B, N, 3, 1]
    
#     # return y_glo.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
#     out = (y_glo + (y_loc - 0.5)).clamp(0, 1).squeeze(-1)                    # [B, N, 3]
#     if is_training:
#         if torch.rand(1).item() < 0.5:
#             return y_glo.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
#     return out.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] # log-radiance
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    
    tone_param: Optional[Float[Tensor, "batch toneparam_len"]] = None
    tone_mapper: Optional[Callable[..., Tensor]] = None   # 或改成 None
    mean_expos: Optional[Float[Tensor, "batch one_len"]] = None
    extra_features: Optional[Float[Tensor, "batch gaussian dim"]] = None
    scene_params: Optional[Float[Tensor, "batch toneparam_len"]] = None
    # levels: Float[Tensor, "batch gaussian"]

# @dataclass
# class HDRGaussians:
#     means: Float[Tensor, "batch gaussian dim"]
#     covariances: Float[Tensor, "batch gaussian dim dim"]
#     harmonics: Float[Tensor, "batch gaussian 3 d_sh"] # log-radiance
#     opacities: Float[Tensor, "batch gaussian"]
#     scales: Float[Tensor, "batch gaussian 3"]
#     rotations: Float[Tensor, "batch gaussian 4"]
#     # levels: Float[Tensor, "batch gaussian"]
