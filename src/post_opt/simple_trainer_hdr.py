import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import imageio
import matplotlib
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import torchvision
import sys
from plyfile import PlyData, PlyElement

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.types import Gaussians
from src.post_opt.datasets.colmap import Dataset
from src.post_opt.datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
# from fused_ssim import fused_ssim

from torchmetrics.functional import structural_similarity_index_measure as ssim_func
def fused_ssim(img1, img2, padding="valid"):
    # torchmetrics 的 ssim 返回值是 0~1，1是最好
    # AnySplat 里的 ssimloss = 1.0 - fused_ssim(...)
    return ssim_func(img1, img2, data_range=1.0)

from src.utils.image import process_image
from src.post_opt.exporter import export_splats
from src.post_opt.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from src.post_opt.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

# from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
# from gsplat.optimizers import SelectiveAdam
# from gsplat.rendering import rasterization
from gsplat import rasterization as gsplat_rasterization, spherical_harmonics
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from src.post_opt.gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.transform import Rotation as R_as_scipy

from src.model.model.anysplat import AnySplat


def rasterization(means, quats, *args, **kwargs):
    """Convert InstantHDR's stored xyzw rotations to gsplat's wxyz order."""
    return gsplat_rasterization(means, quats[..., [3, 0, 1, 2]], *args, **kwargs)


import numpy as np
import cv2
import torch
tonemap_mu = lambda x : (torch.log(torch.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1))
def process_hdr_exr(image_HDR_path):
    import torchvision.transforms.functional as F
    img = cv2.imread(image_HDR_path, cv2.IMREAD_UNCHANGED)[:, :, :3][:, :, ::-1]
    x = torch.from_numpy(img.copy()).permute(2, 0, 1) # [C, H, W]
    x = F.resize(x, [448, 448], antialias=True)
    # x = x / (x.max() + 1e-8) # 防止除零
    x = x / (x.flatten().quantile(0.99).clamp(min=1e-6))
    x = torch.log(x.clamp(0, 1) * 5000.0 + 1.0) / np.log(5000.0 + 1.0)
    return x

def optimize_sim3(src_c2ws, tgt_c2ws, num_iters=10000, lr=1e-2):
    """
    优化 sim(3) 使得 transform(src) ≈ tgt
    src_c2ws: [V, 4, 4] GT poses
    tgt_c2ws: [V, 4, 4] predicted poses (与高斯同坐标系)
    """
    device = src_c2ws.device

    def aa2mat(aa):
        theta = aa.norm() + 1e-8
        k = aa / theta
        K = torch.zeros(3, 3, device=device)
        K[0,1], K[0,2], K[1,0], K[1,2], K[2,0], K[2,1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        return torch.eye(3, device=device) + torch.sin(theta)*K + (1-torch.cos(theta))*(K@K)

    # 提取平移向量 [V, 3]
    src_t = src_c2ws[:, :3, 3]
    tgt_t = tgt_c2ws[:, :3, 3]

    # --- 核心改进：中心对齐 ---
    src_center = src_t.mean(dim=0, keepdim=True)
    tgt_center = tgt_t.mean(dim=0, keepdim=True)
    
    # 初始化参数
    log_s = torch.nn.Parameter(torch.zeros(1, device=device))
    rot = torch.nn.Parameter(torch.zeros(3, device=device))
    t = torch.nn.Parameter(torch.zeros(3, device=device))
    
    # 必须把 log_s 加入优化器！
    # optimizer = torch.optim.Adam([rot, t, log_s], lr=lr)
    optimizer = torch.optim.Adam([
        {'params': [rot],   'lr': lr*0.1},           # 旋转通常需要较小 LR
        {'params': [t],     'lr': lr},           # 平移对初值敏感，可以稍大
        {'params': [log_s], 'lr': lr}            # 尺度缩放建议与平移同步
    ])

    for step in range(num_iters):
        optimizer.zero_grad()
        s = torch.exp(log_s)
        R = aa2mat(rot) # 建议使用更稳定的 Rodrigues 公式实现
        
        # 1. 对旋转进行变换
        aligned_R = R @ src_c2ws[:, :3, :3]
        
        # 2. 对平移进行变换（围绕中心进行旋转缩放）
        # 公式：s * R * (p - center) + t + tgt_center
        centered_src_t = src_t - src_center
        aligned_t = s * (centered_src_t @ R.T) + t + tgt_center
        
        loss_t = F.mse_loss(aligned_t, tgt_t)
        loss_R = F.mse_loss(aligned_R, tgt_c2ws[:, :3, :3])
        
        loss = loss_t + loss_R
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: loss={loss.item():.6f}, loss_t={loss_t.item():.6f}, loss_R={loss_R.item():.6f}, scale={s.item():.4f}")

    return torch.exp(log_s).detach(), aa2mat(rot).detach(), t.detach()

def align_poses_sim3(src_c2ws, tgt_c2ws):
    """
    用 Umeyama 算法求 sim(3) 对齐: tgt = s * R @ src + t
    src_c2ws, tgt_c2ws: [N, 4, 4] c2w matrices
    返回 s, R, t 使得变换后的 src 尽量接近 tgt
    """
    # 提取相机中心 (c2w 的平移部分)
    src_pts = src_c2ws[:, :3, 3]  # [N, 3]
    tgt_pts = tgt_c2ws[:, :3, 3]  # [N, 3]

    # 去中心化
    src_mean = src_pts.mean(dim=0)
    tgt_mean = tgt_pts.mean(dim=0)
    src_centered = src_pts - src_mean
    tgt_centered = tgt_pts - tgt_mean

    # 求 scale
    src_var = (src_centered ** 2).sum()
    
    # 求旋转 (SVD)
    H = src_centered.T @ tgt_centered  # [3, 3]
    U, S, Vt = torch.linalg.svd(H)
    
    # 处理 reflection
    d = torch.det(Vt.T @ U.T)
    sign_mat = torch.diag(torch.tensor([1, 1, d.sign()], device=src_c2ws.device, dtype=src_c2ws.dtype))
    
    R = Vt.T @ sign_mat @ U.T  # [3, 3]
    scale = (S * sign_mat.diag()).sum() / src_var
    t = tgt_mean - scale * R @ src_mean

    return scale, R, t

def apply_sim3_to_c2ws(c2ws, scale, R, t):
    """
    对一组 c2w 矩阵应用 sim(3) 变换
    c2ws: [N, 4, 4]
    """
    aligned = c2ws.clone()
    # 变换旋转部分: R_new = R @ R_old
    aligned[:, :3, :3] = R.unsqueeze(0) @ c2ws[:, :3, :3]
    # 变换平移部分: t_new = s * R @ t_old + t
    aligned[:, :3, 3] = scale * (R.unsqueeze(0) @ c2ws[:, :3, 3].unsqueeze(-1)).squeeze(-1) + t
    return aligned


def convert_intrinsics(meta_data):
    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(fx) / float(store_w)
    intrinsics[1, 1] = float(fy) / float(store_h)
    intrinsics[0, 2] = float(cx) / float(store_w)
    intrinsics[1, 2] = float(cy) / float(store_h)
    return intrinsics

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

#     # # 两层 MLP（逐通道，batch）
#     # log_rad = F.relu(log_rad * W1[:, None] + b1[:, None])                       # [B,N,3,h]
#     # log_rad = (log_rad * W2[:, None]).sum(-1, keepdim=True) + b2[:, None]       # [B,N,3,1]
#     # rgb = torch.sigmoid(log_rad).squeeze(-1)                              # [B,N,3]

#     # --- 分块处理逻辑 ---
#     chunk_size = 1000000  # 根据显存大小调整这个值 (比如 10^6)
#     log_rad_chunks = torch.split(log_rad, chunk_size, dim=1)
#     rgb_chunks = []
#     for chunk in log_rad_chunks:
#         hidden = F.relu(chunk * W1[:, None] + b1[:, None])          # [B, chunk_N, 3, h]
#         out = (hidden * W2[:, None]).sum(-1, keepdim=True) + b2[:, None] # [B, chunk_N, 3, 1]
#         rgb_chunks.append(torch.sigmoid(out))
#     rgb = torch.cat(rgb_chunks, dim=1).squeeze(-1)
        
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

    # # --- Local Tone Mapping (5 -> h -> 1) ---
    # # 拼接输入: RGB(1) + Feat(4) = 5
    # f_exp = f.unsqueeze(2).expand(-1, -1, C, -1)                                # [B, N, 3, 4]
    # x_loc = torch.cat([x, f_exp], dim=-1)                           # [B, N, 3, 5]

    # w1_dim = (1 + F_dim) * hidden
    # w1, b1, w2, b2 = p_l.view(B, 1, C, -1).split([w1_dim, hidden, hidden, 1], dim=-1)
    # w1 = w1.view(B, 1, C, 1 + F_dim, hidden)                                    # [B, 1, 3, 5, h]
    # h = F.relu(torch.matmul(x_loc.unsqueeze(-2), w1).squeeze(-2) + b1)          # [B, N, 3, h]
    # y_loc = torch.sigmoid((h * w2).sum(-1, keepdim=True) + b2)                  # [B, N, 3, 1]

    # --- 融合与还原 ---
    return y_glo.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
    out = (y_glo + (y_loc - 0.5)).clamp(0, 1).squeeze(-1)                         # [B, N, 3]
    if is_training:
        if torch.rand(1).item() < 0.5:
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


# pytorch3d/pytorch3d/transforms/rotation_conversions.py at main · facebookresearch/pytorch3d
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

def colorize_depth_maps(depth_map, min_depth=0.0, max_depth=1.0, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]
    
    # colorize
    cm = matplotlib.colormaps[cmap]
    # depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    depth = ((depth - depth.min()) / (depth.max() - depth.min())).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Feed-forward InstantHDR initialization checkpoint.
    checkpoint: Optional[str] = None
    # Evaluate exposure-conditioned LDR or tone-mapped HDR.
    eval_mode: Literal["ldr", "hdr"] = "ldr"
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # Images are already in memory; workers can be enabled explicitly.
    num_workers: int = 0
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 3_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [1, 1_000, 3_000, 7_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [1, 1_000, 3_000, 7_000, 10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [1, 1_000, 3_000, 7_000, 10_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False
    
    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 4
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 1e-10
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "vgg"

    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh: float = 2.5e-3

    # lr_means: float = 0
    # lr_scales: float = 0
    # lr_quats: float = 0
    # lr_opacities: float = 0
    # lr_sh: float = 0

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            # strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            # strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            # strategy.reset_every = int(strategy.reset_every * factor)
            # strategy.refine_every = int(strategy.refine_every * factor)

            strategy.refine_start_iter = 30000
            strategy.refine_stop_iter = 0
            strategy.reset_every = 30000
            strategy.refine_every = 30000

            # 568811
            # strategy.refine_start_iter = 100
            # strategy.refine_stop_iter = 700
            # strategy.reset_every = 300
            # strategy.refine_every = 50

        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

# !!!!
def create_splats_with_optimizers(
    gaussians: Gaussians,
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    cfg: Config = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    # TODO
    points = gaussians.means[0].detach().float()
    scales = torch.log(gaussians.scales[0].detach().float())
    quats = gaussians.rotations[0].detach().float()
    opacities = torch.logit(gaussians.opacities[0].detach().float())
    harmonics = gaussians.harmonics[0].detach().float().permute(0, 2, 1).contiguous()
    
    tone_param = gaussians.tone_param[0].detach().float()
    mean_expos = gaussians.mean_expos[0].detach().float()
    extra_features = gaussians.extra_features[0].detach().float()
    # scene_params = gaussians.scene_params[0].detach().float()

    N = points.shape[0]
    
    scene_scale = 1.0
    print(f"[Filter] Before mask: {N} gaussians")
    print(f"[Filter] Opacity stats: min={opacities.sigmoid().min().item():.4f}, max={opacities.sigmoid().max().item():.4f}, mean={opacities.sigmoid().mean().item():.4f}")
    masks = opacities.sigmoid() > 0.01
    print(f"[Filter] After mask (>0.01): {masks.sum().item()} gaussians kept, {(~masks).sum().item()} removed ({(~masks).sum().item()/N*100:.1f}%)")
    # masks = opacities.sigmoid() > 0.01
    harmonics = harmonics[masks]
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points[masks]), cfg.lr_means * scene_scale), # 0.00016
        ("scales", torch.nn.Parameter(scales[masks]), cfg.lr_scales), # 0.005
        ("quats", torch.nn.Parameter(quats[masks]), cfg.lr_quats), # 0.001
        ("opacities", torch.nn.Parameter(opacities[masks]), cfg.lr_opacities), # 0.05

        ("tone_param", torch.nn.Parameter(tone_param), cfg.lr_sh * 0.1), # 2.5e-3
        ("mean_expos", torch.nn.Parameter(mean_expos), 0),
        ("extra_features", torch.nn.Parameter(extra_features[masks]), 0),
        # ("scene_params", torch.nn.Parameter(scene_params), cfg.lr_sh),
        # ("extra_features", torch.nn.Parameter(extra_features[masks]), 0),
    ]
    
    params.append(("sh0", torch.nn.Parameter(harmonics[:, :1, :]), cfg.lr_sh))
    params.append(("shN", torch.nn.Parameter(harmonics[:, 1:, :]), cfg.lr_sh/20))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""
    def visualize_pose_comparison(self, gt_images, c2ws_A, c2ws_B, Ks, exposures,
                                save_dir, names=("sim3_aligned", "pred_pose")):
        """左GT | 中c2ws_A | 右c2ws_B 复用训练管线保证一致"""
        device = self.device
        V, _, h, w = gt_images.shape
        os.makedirs(save_dir, exist_ok=True)

        def render_one(c2w_single, K_single, exp_val):
            c2w = c2w_single[None].float().to(device)
            K_px = K_single[None].float().to(device).clone()
            K_px[:, 0, :] *= w
            K_px[:, 1, :] *= h

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=c2w, Ks=K_px, width=w, height=h,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane,
                render_mode="RGB+ED",
            )
            # 和 eval 完全一致的 tonemapping
            colors = renders[..., :3]        # [1, H, W, 3]
            local_feat = renders[..., 3:-1]  # [1, H, W, 4]

            colors = colors.permute(0, 3, 1, 2)[None, ...]       # [1, 1, 3, H, W]
            local_feat = local_feat.permute(0, 3, 1, 2)[None, ...]  # [1, 1, 4, H, W]
            exp_t = torch.log2(exp_val.to(device).to(colors)).detach()
            rad = torch.log2(colors + 1e-6) + (exp_t - self.splats['mean_expos']).to(colors)
            colors = tonemapping(rad, local_feat, params=self.splats['tone_param'][None, ...], is_training=False)
            # colors = tonemapping(rad, params=self.splats['tone_param'][None, ...])
            colors = colors[0, 0].clamp(0, 1)  # [3, H, W]

            # debug: alpha 统计
            alpha_mean = alphas.mean().item()
            return colors, alpha_mean

        with torch.no_grad():
            for idx in range(V):
                gt_vis = gt_images[idx].float().to(device).clamp(0, 1)

                img_A, alpha_A = render_one(c2ws_A[idx].to(device), Ks[idx], exposures[idx])
                img_B, alpha_B = render_one(c2ws_B[idx].to(device), Ks[idx], exposures[idx])

                print(f"  View {idx}: alpha_A={alpha_A:.4f}, alpha_B={alpha_B:.4f}")
                print(f"    pos_A={c2ws_A[idx, :3, 3].tolist()}")
                print(f"    pos_B={c2ws_B[idx, :3, 3].tolist()}")

                canvas = torch.cat([gt_vis, img_A, img_B], dim=2).cpu()
                torchvision.utils.save_image(canvas, f"{save_dir}/{idx:02d}.png")
        print(f"Saved {V} comparisons to {save_dir}")

    def optimize_splats_by_rendering(self, gt_c2ws, Ks, gt_images, 
                                     exposures, optimizer=None, num_iters=1000, lr_scale=1.0):
        """Pose已对齐, 冻结pose优化splat本身"""
        device = self.device
        V = gt_c2ws.shape[0]
        h, w = gt_images.shape[-2], gt_images.shape[-1]
        gt_c2ws = gt_c2ws.detach().float().to(device)
        Ks_px = Ks.detach().float().to(device).clone()
        Ks_px[:, 0, :] *= w
        Ks_px[:, 1, :] *= h

        # 确保splats可训练
        for p in self.splats.parameters():
            p.requires_grad_(True)
        # mean_expos 冻结
        self.splats['mean_expos'].requires_grad_(False)

        # 构建optimizer
        param_groups = [
            {'params': self.splats['means'],     'lr': 1.6e-4 * lr_scale, 'name': 'means'},
            {'params': self.splats['scales'],     'lr': 5e-3 * lr_scale, 'name': 'scales'},
            {'params': self.splats['quats'],      'lr': 1e-3 * lr_scale, 'name': 'quats'},
            {'params': self.splats['opacities'],  'lr': 5e-2 * lr_scale, 'name': 'opacities'},
            {'params': self.splats['sh0'],        'lr': 2.5e-3 * lr_scale, 'name': 'sh0'},
            {'params': self.splats['shN'],        'lr': 1.25e-4 * lr_scale, 'name': 'shN'},
            {'params': self.splats['tone_param'], 'lr': 2.5e-4 * lr_scale, 'name': 'tone_param'},
        ]
        optimizer = torch.optim.Adam(param_groups, eps=1e-15)

        for step in range(num_iters):
            optimizer.zero_grad()
            total_loss = 0.0

            # 随机采样一部分view或全部
            indices = list(range(V))
            for idx in indices:
                renders, _, _ = self.rasterize_splats(
                    camtoworlds=gt_c2ws[idx:idx+1],
                    Ks=Ks_px[idx:idx+1],
                    width=w, height=h,
                    sh_degree=self.cfg.sh_degree,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    render_mode="RGB+ED",
                )
                colors = renders[..., :3].permute(0, 3, 1, 2)[None]

                exp_t = torch.log2(exposures[idx].to(device).to(colors)).detach()
                rad = torch.log2(colors + 1e-6) + (exp_t - self.splats['mean_expos']).to(colors)
                colors = tonemapping(rad, self.splats['tone_param'][None])
                colors_p = colors[0].clamp(0, 1)
                gt_p = gt_images[idx:idx+1].to(device)

                ssim_loss = 1.0 - fused_ssim(colors_p.float(), gt_p.float(), padding="valid")
                l1_loss = F.l1_loss(colors_p.float(), gt_p.float())
                loss = 0.2 * ssim_loss + 0.8 * l1_loss
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            if step % 50 == 0:
                print(f"  Splat opt step {step}: loss={total_loss/V:.6f}")

        # 返回优化后的splats dict (detached)
        optimized = {k: v.detach().clone() for k, v in self.splats.items()}
        return optimized
    
    def optimize_sim3_by_rendering(self, gt_c2ws, pred_c2ws, Ks, gt_images, exposures,
                                    num_iters=200, lr=1e-3):
        """用Umeyama初始化sim3 再用渲染loss微调"""
        device = self.device
        V = gt_c2ws.shape[0]
        h, w = gt_images.shape[-2], gt_images.shape[-1]
        gt_c2ws = gt_c2ws.detach().float().to(device)
        pred_c2ws = pred_c2ws.detach().float().to(device)
        Ks_px = Ks.detach().float().to(device).clone()
        Ks_px[:, 0, :] *= w
        Ks_px[:, 1, :] *= h

        # Umeyama 初始化
        from scipy.spatial.transform import Rotation as R_scipy
        s_init, R_init, t_init = align_poses_sim3(gt_c2ws, pred_c2ws)
        aa_init = torch.from_numpy(
            R_scipy.from_matrix(R_init.detach().cpu().numpy()).as_rotvec()
        ).float().to(device)

        log_s = torch.nn.Parameter(torch.log(s_init).reshape(1).to(device))
        rot_aa = torch.nn.Parameter(aa_init)
        t_sim3 = torch.nn.Parameter(t_init.to(device))

        optimizer = torch.optim.Adam([
            {'params': log_s, 'lr': lr},
            {'params': rot_aa, 'lr': lr},
            {'params': t_sim3, 'lr': lr},
        ])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=max(num_iters, 1), eta_min=lr * 0.01
        # )

        def aa_to_mat(aa):
            theta = aa.norm().clamp(min=1e-8)
            k = aa / theta
            K0 = torch.zeros(3, 3, device=device)
            K0[0, 1] = -k[2]; K0[0, 2] = k[1]
            K0[1, 0] = k[2];  K0[1, 2] = -k[0]
            K0[2, 0] = -k[1]; K0[2, 1] = k[0]
            return torch.eye(3, device=device) + torch.sin(theta) * K0 + (1 - torch.cos(theta)) * (K0 @ K0)

        # 冻结splats
        for p in self.splats.parameters():
            p.requires_grad_(False)

        for step in range(num_iters):
            optimizer.zero_grad()
            total_loss = 0.0

            for idx in range(V):
                s = torch.exp(log_s)
                R_mat = aa_to_mat(rot_aa)
                c2w_i = torch.zeros(1, 4, 4, device=device)
                c2w_i[0, :3, :3] = R_mat @ gt_c2ws[idx, :3, :3]
                c2w_i[0, :3, 3] = s * (R_mat @ gt_c2ws[idx, :3, 3]) + t_sim3
                c2w_i[0, 3, 3] = 1.0

                renders, _, _ = self.rasterize_splats(
                    camtoworlds=c2w_i, Ks=Ks_px[idx:idx+1],
                    width=w, height=h, sh_degree=self.cfg.sh_degree,
                    near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane,
                    render_mode="RGB+ED",
                )
                colors = renders[..., :3].permute(0, 3, 1, 2)[None]
                feat = renders[..., 3:-1].permute(0, 3, 1, 2)[None]
                exp_t = torch.log2(exposures[idx].to(device).to(colors)).detach()
                rad = torch.log2(colors + 1e-6) + (exp_t - self.splats['mean_expos']).to(colors)
                colors = tonemapping(rad, feat, self.splats['tone_param'][None], is_training=False)
                # colors = tonemapping(rad, self.splats['tone_param'][None])
                colors_p = colors[0].clamp(0, 1)       # [1, 3, H, W]
                gt_p = gt_images[idx:idx+1].to(device)  # [1, 3, H, W]
                
                # import torch
                # import torchvision
                # comparison_batch = torch.cat([gt_p,colors_p], dim=0)
                # torchvision.utils.save_image(comparison_batch, 'test.jpg', nrow=2, normalize=True)

                ssim_loss = 1.0 - fused_ssim(colors_p.float(), gt_p.float(), padding="valid")
                l1_loss = F.l1_loss(colors_p.float(), gt_p.float())
                loss = 0.8 * ssim_loss + 0.2 * l1_loss
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            # scheduler.step()
            print(f"  Sim3 step {step}: loss={total_loss:.6f}, scale={torch.exp(log_s).item():.4f}")

        # 恢复splats梯度
        for p in self.splats.parameters():
            p.requires_grad_(True)
            p.grad = None

        with torch.no_grad():
            s_final = torch.exp(log_s)
            R_final = aa_to_mat(rot_aa)
            t_final = t_sim3.clone()
        return s_final, R_final, t_final

    def optimize_poses_by_rendering(self, pred_c2ws, Ks, gt_images, exposures,
                                    num_iters=200, lr=1e-3):
        """直接优化每个视角的外参 c2w"""
        device = self.device
        V = pred_c2ws.shape[0]
        h, w = gt_images.shape[-2], gt_images.shape[-1]
        pred_c2ws = pred_c2ws.detach().float().to(device)
        Ks_px = Ks.detach().float().to(device).clone()
        Ks_px[:, 0, :] *= w
        Ks_px[:, 1, :] *= h

        # 用 pred_c2ws 初始化，用 axis-angle + t 参数化每个相机
        from scipy.spatial.transform import Rotation as R_scipy

        rot_aas = []
        trans = []
        for i in range(V):
            R_np = pred_c2ws[i, :3, :3].cpu().numpy()
            aa = R_scipy.from_matrix(R_np).as_rotvec()
            rot_aas.append(torch.nn.Parameter(
                torch.from_numpy(aa).float().to(device)))
            trans.append(torch.nn.Parameter(
                pred_c2ws[i, :3, 3].clone().to(device)))

        optimizer = torch.optim.Adam([
            {'params': rot_aas, 'lr': lr},
            {'params': trans,   'lr': lr},
        ])

        def aa_to_mat(aa):
            theta = aa.norm().clamp(min=1e-8)
            k = aa / theta
            K0 = torch.zeros(3, 3, device=device)
            K0[0, 1] = -k[2]; K0[0, 2] =  k[1]
            K0[1, 0] =  k[2]; K0[1, 2] = -k[0]
            K0[2, 0] = -k[1]; K0[2, 1] =  k[0]
            return torch.eye(3, device=device) + torch.sin(theta) * K0 + (1 - torch.cos(theta)) * (K0 @ K0)

        # 冻结 splats
        for p in self.splats.parameters():
            p.requires_grad_(False)

        for step in range(num_iters):
            optimizer.zero_grad()
            total_loss = 0.0

            for idx in range(V):
                R_mat = aa_to_mat(rot_aas[idx])
                c2w_i = torch.zeros(1, 4, 4, device=device)
                c2w_i[0, :3, :3] = R_mat
                c2w_i[0, :3,  3] = trans[idx]
                c2w_i[0,  3,  3] = 1.0

                renders, _, _ = self.rasterize_splats(
                    camtoworlds=c2w_i, Ks=Ks_px[idx:idx+1],
                    width=w, height=h, sh_degree=self.cfg.sh_degree,
                    near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane,
                    render_mode="RGB+ED",
                    open_cov=True,
                )
                colors = renders[..., :3].permute(0, 3, 1, 2)[None]
                feat   = renders[..., 3:-1].permute(0, 3, 1, 2)[None]
                exp_t  = torch.log2(exposures[idx].to(device).to(colors)).detach()
                rad    = torch.log2(colors + 1e-6) + (exp_t - self.splats['mean_expos']).to(colors)
                colors = tonemapping(rad, feat, self.splats['tone_param'][None], is_training=False)
                colors_p = colors[0].clamp(0, 1)
                gt_p     = gt_images[idx:idx+1].to(device)

                ssim_loss = 1.0 - fused_ssim(colors_p.float(), gt_p.float(), padding="valid")
                l1_loss   = F.l1_loss(colors_p.float(), gt_p.float())
                loss = 0.8 * ssim_loss + 0.2 * l1_loss
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            print(f" Test Time Pose Optimization: step {step}, loss={total_loss:.6f}")

        # 恢复 splats 梯度
        for p in self.splats.parameters():
            p.requires_grad_(True)
            p.grad = None

        # 组装优化后的 c2ws
        optimized_c2ws = torch.zeros(V, 4, 4, device=device)
        with torch.no_grad():
            for i in range(V):
                R_mat = aa_to_mat(rot_aas[i])
                optimized_c2ws[i, :3, :3] = R_mat
                optimized_c2ws[i, :3,  3] = trans[i]
                optimized_c2ws[i,  3,  3] = 1.0

        return optimized_c2ws

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # first get the initial 3DGS and camera poses
        from src.runtime import load_model
        model = load_model(cfg.checkpoint, device=self.device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
                
        image_folder = str(Path(cfg.data_dir).resolve())
        image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        images = [process_image(img_path) for img_path in image_names]

        # import re
        # hdr_images = []
        # for img_path in image_names:
        #     hdr_path = img_path.replace('/images/', '/images_hdr/')
        #     pattern = r'_(?:r|ldr)_(\d+)_(\d+)\.(?:jpg|png)$'
        #     def replace_func(match):
        #         num1 = int(match.group(1)) # 倒数第二个数字
        #         num2 = match.group(2)      # 最后一个数字
        #         return f"_hdr_{num1:03d}.exr"
        #     hdr_path = re.sub(pattern, replace_func, hdr_path)
        #     hdr_images.append(process_hdr_exr(hdr_path))
            
        # ctx_indices = [idx for idx, name in enumerate(image_names) if idx % cfg.test_every != 0]
        # tgt_indices = [idx for idx, name in enumerate(image_names) if idx % cfg.test_every == 0]

        # view_groups = {}
        # for idx, path in enumerate(image_names):
        #     name = os.path.basename(path).lower()
        #     if 'train' in name:
        #         view_id = name.split('_')[-2]                       # 提取视角值
        #         exp_time = name.split('_')[-1].split('.')[0]        # 提取曝光度
        #         if exp_time in ['0','2','4']:
        #             view_groups.setdefault(view_id, []).append(idx)
        # # ctx_indices = [indices[random.randint(0, 2)] for indices in view_groups.values()]
        # ctx_indices = [87, 94, 95, 100, 105, 114, 117, 124, 127, 130, 137, 140, 147, 154, 159, 162, 165, 174]

        ctx_indices = [idx for idx, name in enumerate(image_names) if 'train' in os.path.basename(name).lower()]
        tgt_indices = [idx for idx, name in enumerate(image_names) if 'test' in os.path.basename(name).lower()]
        
        if len(ctx_indices) < 2 or not tgt_indices:
            raise ValueError("Expected at least two train* images and one test* image in --data-dir.")
        ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
        tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
        ctx_images = (ctx_images+1)*0.5
        tgt_images = (tgt_images+1)*0.5

        # hdr_tgt_images = torch.stack([hdr_images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)

        # Loading Exposure Time
        exposure_json_path = os.path.join('/'.join(image_folder.split('/')[:-1]), "exposure.json")
        with open(exposure_json_path, 'r') as f:
            exposure_dict = json.load(f)
        for name in image_names:
            value = float(exposure_dict[Path(name).name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid exposure for {name}: {value}")
        ctx_exposure = []
        for idx in ctx_indices:
            ctx_exposure.append(torch.tensor([exposure_dict[os.path.basename(image_names[idx])]],device=ctx_images.device,dtype=ctx_images.dtype))
        tgt_exposure = []
        for idx in tgt_indices:
            tgt_exposure.append(torch.tensor([exposure_dict[os.path.basename(image_names[idx])]],device=ctx_images.device,dtype=ctx_images.dtype))
        
        # TODO
        ctx_idx_list = list(range(len(ctx_indices)))
        # ctx_idx_list = random.sample(range(18), 4)
        # ctx_idx_list = [0, 11, 4, 7]

        for idx in [ctx_indices[i] for i in ctx_idx_list]:
            print(os.path.basename(image_names[idx]))
            print(idx)

        ctx_images = ctx_images[:,ctx_idx_list]
        ctx_exposure = [ctx_exposure[i] for i in ctx_idx_list]
        tgt_images = tgt_images[:,:].to(torch.bfloat16)
        tgt_exposure = tgt_exposure[:]
        b, v, _, h, w = tgt_images.shape
        print(ctx_images.shape)
        print(ctx_exposure)

        # run inference
        encoder_output = model.encoder(
            ctx_images,
            ctx_exposure,
            global_step=0,
            visualization_dump={},
        )
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        
        encoder_output = model.encoder(
            ctx_images[:,:],
            ctx_exposure[:],
            global_step=0,
            visualization_dump={},
        )
        _, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        
        num_context_view = ctx_images.shape[1]
        vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(vggt_input_image, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        with torch.cuda.amp.autocast(enabled=False):
            fp32_tokens = [token.float() for token in aggregated_tokens_list]
            pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1] # [B,V,9]
            pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])

        # # 逐张预测target pose，避免大batch导致不准
        # print('====================== Each Target Pose =======================')
        # tgt_pose_encs = []
        # for t_idx in range(tgt_images.shape[1]):
        #     vggt_input = torch.cat((ctx_images, tgt_images[:, t_idx:t_idx+1]), dim=1).to(torch.bfloat16)
        #     with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        #         agg_tokens, _ = model.encoder.aggregator(vggt_input, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        #     with torch.cuda.amp.autocast(enabled=False):
        #         fp32_tokens = [t.float() for t in agg_tokens]
        #         pose_enc = model.encoder.camera_head(fp32_tokens)[-1]  # [B, V_ctx+1, 9]
        #     tgt_pose_encs.append(pose_enc[:, -1:])  # 只取最后一个（target）
        #     print(f'====================== Predict {t_idx} Target Pose =======================')
        # # ctx pose 用第一次的结果（或单独跑一次纯ctx）
        # vggt_ctx = ctx_images.to(torch.bfloat16)
        # with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        #     agg_tokens, _ = model.encoder.aggregator(vggt_ctx, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        # with torch.cuda.amp.autocast(enabled=False):
        #     fp32_tokens = [t.float() for t in agg_tokens]
        #     ctx_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]  # [B, V_ctx, 9]
        # pred_all_pose_enc = torch.cat([ctx_pose_enc] + tgt_pose_encs, dim=1)  # [B, V_ctx+V_tgt, 9]
        # pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])
        # print('====================== Finish Target Pose =======================')

        extrinsic_padding = torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
        pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

        pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
        pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
        pred_all_context_extrinsic, pred_all_target_extrinsic = pred_all_extrinsic[:, :num_context_view], pred_all_extrinsic[:, num_context_view:]
        pred_all_context_intrinsic, pred_all_target_intrinsic = pred_all_intrinsic[:, :num_context_view], pred_all_intrinsic[:, num_context_view:]

        scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / pred_all_context_extrinsic[:, :, :3, 3].mean()
        pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
        pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
        print("scale_factor:", scale_factor)

        ############################## DEBUG ##############################
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            gaussians=gaussians,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            cfg=cfg,
        )
        # print("Model initialized. Number of GS:", len(self.splats["means"]))
        # self.visualize_pose_comparison(
        #     gt_images=ctx_images[0],
        #     c2ws_A=pred_all_context_extrinsic[0],
        #     c2ws_B=pred_context_pose['extrinsic'][0],
        #     Ks=pred_all_context_intrinsic[0],
        #     exposures=ctx_exposure,
        #     save_dir=f"post_vis/pose_cmp",
        # )

        # ########## 7777
        # ########## Optimize predicted poses, then align GT poses to optimized space
        # K_mean = pred_context_pose['intrinsic'].mean(dim=1, keepdim=True)  # [B, 1, 3, 3]
        # pred_all_context_intrinsic = K_mean.expand_as(pred_context_pose['intrinsic'])
        # pred_all_target_intrinsic = K_mean.expand(b, len(tgt_indices), 3, 3)
        # s, R, t = self.optimize_sim3_by_rendering(
        #     ctx_extrinsics[0], pred_context_pose['extrinsic'][0],
        #     pred_all_context_intrinsic[0], ctx_images[0], ctx_exposure,
        #     num_iters=200, lr=1e-3,
        # )
        # pred_all_context_extrinsic = apply_sim3_to_c2ws(ctx_extrinsics[0].to(device), s, R, t).unsqueeze(0)
        # pred_all_target_extrinsic = apply_sim3_to_c2ws(tgt_extrinsics[0].to(device), s, R, t).unsqueeze(0)

        # ########## 8888
        # ########## Optimize predicted poses, then align GT poses to optimized space
        # # K_mean = pred_all_intrinsic.mean(dim=1, keepdim=True)  # [B, 1, 3, 3]
        # # pred_all_context_intrinsic = K_mean.expand_as(pred_context_pose['intrinsic'])
        # # pred_all_target_intrinsic = K_mean.expand(b, len(tgt_indices), 3, 3)
        # s, R, t = self.optimize_sim3_by_rendering(
        #     pred_all_context_extrinsic[0], pred_context_pose['extrinsic'][0],
        #     pred_all_context_intrinsic[0], ctx_images[0], ctx_exposure,
        #     num_iters=200, lr=1e-3,
        # )
        # pred_all_context_extrinsic = apply_sim3_to_c2ws(pred_all_context_extrinsic[0].to(device), s, R, t).unsqueeze(0)
        # pred_all_target_extrinsic = apply_sim3_to_c2ws(pred_all_target_extrinsic[0].to(device), s, R, t).unsqueeze(0)

        # ########## 9999
        # ########## Optimize predicted poses, then align GT poses to optimized space
        # K_mean = pred_all_intrinsic.mean(dim=1, keepdim=True)  # [B, 1, 3, 3]
        # pred_all_context_intrinsic = K_mean.expand_as(pred_context_pose['intrinsic'])
        # pred_all_target_intrinsic = K_mean.expand(b, len(tgt_indices), 3, 3)
        # pred_all_intrinsic = K_mean.expand_as(pred_all_intrinsic)
        # s, R, t = self.optimize_sim3_by_rendering(
        #     torch.concat([ctx_extrinsics,tgt_extrinsics],dim=1)[0], pred_all_extrinsic[0],
        #     pred_all_intrinsic[0], vggt_input_image[0], ctx_exposure + tgt_exposure,
        #     num_iters=200, lr=1e-3,
        # )
        # pred_all_context_extrinsic = apply_sim3_to_c2ws(ctx_extrinsics[0].to(device), s, R, t).unsqueeze(0)
        # pred_all_target_extrinsic = apply_sim3_to_c2ws(tgt_extrinsics[0].to(device), s, R, t).unsqueeze(0)

        ########## XXXXXXXXXX
        ########## Optimize predicted poses, then align GT poses to optimized space
        K_mean = pred_all_intrinsic.mean(dim=1, keepdim=True)  # [B, 1, 3, 3]
        pred_all_context_intrinsic = K_mean.expand_as(pred_context_pose['intrinsic'])
        pred_all_target_intrinsic = K_mean.expand(b, len(tgt_indices), 3, 3)
        pred_all_intrinsic = K_mean.expand_as(pred_all_intrinsic)
        tmp = self.optimize_poses_by_rendering(
            pred_all_extrinsic[0], pred_all_intrinsic[0], 
            vggt_input_image[0], ctx_exposure + tgt_exposure,
            num_iters=100, lr=1e-3,
            # num_iters=20, lr=1e-3,
        )
        tmp = tmp[None,...]
        pred_all_context_extrinsic, pred_all_target_extrinsic = tmp[:, :num_context_view], tmp[:, num_context_view:]


        # ########## XXXXXXXXXX TrainPose
        # ########## Optimize predicted poses, then align GT poses to optimized space
        # K_mean = pred_all_intrinsic.mean(dim=1, keepdim=True)  # [B, 1, 3, 3]
        # pred_all_context_intrinsic = K_mean.expand_as(pred_context_pose['intrinsic'])
        # pred_all_target_intrinsic = K_mean.expand_as(pred_all_target_intrinsic)
        # pred_all_intrinsic = K_mean.expand_as(pred_all_intrinsic)

        # pred_all_context_extrinsic = self.optimize_poses_by_rendering(
        #     pred_all_context_extrinsic[0], pred_all_context_intrinsic[0], 
        #     ctx_images[0], ctx_exposure,
        #     num_iters=200, lr=1e-3,
        # )
        # pred_all_context_extrinsic = pred_all_context_extrinsic[None,...]


        # Load data: Training data should contain initial points and colors.
        # self.parser = Parser(
        #     data_dir=cfg.data_dir,
        #     factor=cfg.data_factor,
        #     normalize=cfg.normalize_world_space,
        #     test_every=cfg.test_every,
        # )
        self.trainset = Dataset(
            split="train",
            images=ctx_images[0].detach().cpu().numpy(),
            exposure=torch.concat(ctx_exposure).detach().cpu().numpy(),
            camtoworlds=pred_all_context_extrinsic[0].detach().cpu().numpy(),
            Ks=pred_all_context_intrinsic[0].detach().cpu().numpy(),
            # camtoworlds=ctx_extrinsics[0].detach().cpu().numpy(),
            # Ks=ctx_intrinsics[0].detach().cpu().numpy(),
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        hdr_targets = None
        if cfg.eval_mode == "hdr":
            import re
            hdr_paths = []
            for idx in tgt_indices:
                name = re.sub(r"_(?:r|ldr)_(\d+)_(\d+)\.(?:png|jpg|jpeg)$",
                              lambda m: f"_hdr_{int(m.group(1)):03d}.exr", Path(image_names[idx]).name)
                hdr_paths.append(Path(image_folder).parent / "images_hdr" / name)
            if not all(p.is_file() for p in hdr_paths):
                raise FileNotFoundError("HDR evaluation requires matching images_hdr/*_hdr_NNN.exr files.")
            hdr_targets = torch.stack([process_hdr_exr(str(p)) for p in hdr_paths]).numpy()
        self.valset = Dataset(
            hdr_images=hdr_targets,
            images=tgt_images[0].detach().cpu().float().numpy(),
            # hdr_images=hdr_tgt_images[0].detach().cpu().float().numpy(),
            exposure=torch.concat(tgt_exposure).detach().cpu().numpy(),
            camtoworlds=pred_all_target_extrinsic[0].detach().cpu().numpy(),
            Ks=pred_all_target_intrinsic[0].detach().cpu().numpy(),
            # camtoworlds=tgt_extrinsics[0].detach().cpu().numpy(),
            # Ks=tgt_intrinsics[0].detach().cpu().numpy(),
            split="val"
        )
        self.val_image_names = [os.path.basename(image_names[i]) for i in tgt_indices]

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            gaussians=gaussians,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            cfg=cfg,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=1.0
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )
        
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        open_cov = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])
        extra = self.splats["extra_features"]  # [N, 4]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        
        cam_centers = camtoworlds[:, :3, 3]
        dirs = means.unsqueeze(0) - cam_centers.unsqueeze(1)
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-6)
        V, N, _ = dirs.shape
        K = colors.shape[1]
        features_expanded = colors.unsqueeze(0).expand(V, -1, -1, -1) # [V, N, K, 3]
        log_rad = spherical_harmonics(
            4,
            dirs.reshape(-1, 3),
            features_expanded.reshape(-1, K, 3)
        ).reshape(V, N, 3)
        linear_rad = torch.exp(log_rad)
        # inputs = linear_rad
        inputs = torch.cat([linear_rad, extra[None,...]], dim=-1)
        kwargs.pop("sh_degree", None)

        covariance = build_covariance(scales[None].detach(), quats[None].detach()).squeeze(0).requires_grad_(False) if open_cov else None
        # covariance = build_covariance(scales[None].detach(), quats[None].detach()).squeeze(0).requires_grad_(False)
        # covariance = None
        # covariance = build_covariance(scales[None].detach(), quats[None].detach()).squeeze(0).requires_grad_(False)
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=inputs,
            covars=covariance,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            radius_clip=0.1,
            backgrounds=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).cuda().unsqueeze(0).repeat(1, 1),
            # backgrounds=torch.tensor([0.0, 0.0, 0.0]).cuda().unsqueeze(0).repeat(1, 1),
            **kwargs,
        )

        # covariance = build_covariance(scales[None], quats[None]).squeeze(0)
        # render_colors, render_alphas, info = rasterization(
        #     means=means,
        #     quats=quats,
        #     scales=scales,
        #     opacities=opacities,
        #     colors=colors,
        #     # covars=covariance,
        #     viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        #     Ks=Ks,  # [C, 3, 3]
        #     width=width,
        #     height=height,
        #     packed=self.cfg.packed,
        #     absgrad=(
        #         self.cfg.strategy.absgrad
        #         if isinstance(self.cfg.strategy, DefaultStrategy)
        #         else False
        #     ),
        #     sparse_grad=self.cfg.sparse_grad,
        #     rasterize_mode=rasterize_mode,
        #     distributed=self.world_size > 1,
        #     camera_model=self.cfg.camera_model,
        #     radius_clip=0.1,
        #     backgrounds=torch.tensor([0.0, 0.0, 0.0]).cuda().unsqueeze(0).repeat(1, 1),
        #     **kwargs,
        # )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                eval_tic = time.time()
                if step == 0:
                    self.eval(step, open_cov=True)
                else:
                    self.eval(step)
                    # self.render_traj(step)
                torch.cuda.synchronize()
                eval_time = time.time() - eval_tic
                global_tic += eval_time

            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            # sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            sh_degree_to_use = cfg.sh_degree

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            elif renders.shape[-1] == 7:
                colors, local_feat = renders[..., 0:3], renders[..., 3:]
                depths = None
            else:
                colors, depths = renders, None

            # Color ToneMapping:
            colors = colors.permute(0,3,1,2)[None,...]
            local_feat = local_feat.permute(0,3,1,2)[None,...]
            rad_time = torch.log2(colors + 1e-6) + (torch.log2(data['exposure']).to(colors) - self.splats.mean_expos).to(colors)
            colors = tonemapping(rad_time, local_feat, params=self.splats.tone_param[None,...], is_training=True)
            # colors = tonemapping(rad_time, params=self.splats.tone_param[None,...])
            
            colors = colors[0].permute(0,2,3,1)

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            
            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]
                    # shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]

                # export_splats(
                #     means=means,
                #     scales=scales,
                #     quats=quats,
                #     opacities=opacities,
                #     sh0=sh0,
                #     shN=shN,
                #     format="ply",
                #     save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                # )
                export_ply(
                    means=means,
                    scales=scales,
                    rotations=quats,
                    harmonics=torch.cat([sh0, shN], dim=1).permute(0, 2, 1),
                    opacities=opacities.sigmoid(),
                    path=Path(f"{self.ply_dir}/point_cloud_{step}.ply"),
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            
            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Save the updated splats together with their evaluation cameras.
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict(),
                        "val_image_names": self.val_image_names,
                        "val_camtoworlds": torch.from_numpy(self.valset.camtoworlds.copy()),
                        "val_Ks": torch.from_numpy(self.valset.Ks.copy())}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # # eval the full set
            # if step in [i - 1 for i in cfg.eval_steps]:
            #     self.eval(step)
            #     # self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", open_cov=False):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            # pixels = data["hdr_image"].to(device) # HDR
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            render_colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                # radius_clip=0.1,
                render_mode="RGB+ED",
                masks=masks,
                open_cov=open_cov,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = render_colors[..., :3]
            local_feat = render_colors[..., 3:-1]
            depths = render_colors[..., -1]

            colors = colors.permute(0, 3, 1, 2)[None, ...]
            if cfg.eval_mode == "ldr":
                local_feat = local_feat.permute(0, 3, 1, 2)[None, ...]
                rad_time = torch.log2(colors + 1e-6) + (
                    torch.log2(data["exposure"]).to(colors) - self.splats.mean_expos
                )
                colors = tonemapping(rad_time, local_feat, params=self.splats.tone_param[None, ...], is_training=False)
            else:
                pixels = data["hdr_image"].to(device)
                colors = tonemap_mu((colors / colors.flatten(-3).quantile(0.99, dim=-1)
                                     .reshape(*colors.shape[:-3], 1, 1, 1).clamp(min=1e-6)).clamp(max=1))

            colors = colors[0].permute(0,2,3,1)
            # colors = colors + (1.0 - alphas) * 0.5

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # # write images
                # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                # canvas = (canvas * 255).astype(np.uint8)
                # imageio.imwrite(
                #     f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                #     canvas,
                # )
                # torchvision.utils.save_image(pixels.permute(0, 3, 1, 2), f"{self.render_dir}/gt_rgb_{stage}_step{step}_{i:04d}.png")
                # torchvision.utils.save_image(colors.permute(0, 3, 1, 2), f"{self.render_dir}/render_rgb_{stage}_step{step}_{i:04d}.png")
                # # save depth & normal map

                name_stem = os.path.splitext(self.val_image_names[i])[0]
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{name_stem}.png",
                    canvas,
                )
                torchvision.utils.save_image(pixels.permute(0, 3, 1, 2), f"{self.render_dir}/gt_rgb_step{step}_{name_stem}.png")
                torchvision.utils.save_image(colors.permute(0, 3, 1, 2), f"{self.render_dir}/render_rgb_step{step}_{name_stem}.png")
                # Diff Heatmap
                diff = (colors - pixels).abs().mean(dim=-1).squeeze().cpu()  # [H,W]
                diff_color = colorize_depth_maps(diff / (diff.max() + 1e-6), cmap="turbo")
                torchvision.utils.save_image(diff_color.float(), f"{self.render_dir}/diff_step{step}_{name_stem}.png")
                

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            # radius_clip=0.1,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    if world_size != 1:
        raise ValueError("Run post-optimization with one visible GPU per scene.")
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        if len(cfg.ckpt) != 1:
            raise ValueError("Pass one checkpoint for this scene.")
        saved = torch.load(cfg.ckpt[0], map_location=runner.device, weights_only=True)
        runner.splats.load_state_dict(saved["splats"], strict=True)
        if "val_image_names" in saved:
            if saved["val_image_names"] != runner.val_image_names:
                raise ValueError("Checkpoint evaluation images do not match --data-dir.")
            runner.valset.camtoworlds = saved["val_camtoworlds"].cpu().numpy()
            runner.valset.Ks = saved["val_Ks"].cpu().numpy()
        if cfg.pose_opt and "pose_adjust" in saved:
            runner.pose_adjust.load_state_dict(saved["pose_adjust"])
        if cfg.app_opt and "app_module" in saved:
            runner.app_module.load_state_dict(saved["app_module"])
        step = saved["step"]
        runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.eval(step=runner.cfg.max_steps)
        # runner.render_traj(step=runner.cfg.max_steps)
    runner.writer.close()
    print("Training complete.")
    # runner.viewer.complete()
    # if not cfg.disable_viewer:
    #     print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
