"""
Early Fusion InnerModel
=======================
Concatenates front RGB + tactile images along the channel dimension BEFORE
the first conv layer, so the model sees 6 channels per frame instead of 3.

Key differences from the original single-modality model:
  - img_channels = 6  (3 RGB + 3 tactile)
  - act_emb: Linear(7, ...) for 7-DoF robot actions
  - conv_in: (num_steps_conditioning + 1) * 6 input channels
  - conv_out: 6 output channels (predicts both RGB and tactile next frame)
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int                  # 6 for early fusion (3 RGB + 3 tactile)
    num_steps_conditioning: int        # number of past frames used as context
    cond_channels: int                 # dimension of conditioning embedding
    depths: List[int]                  # number of ResBlocks per UNet level
    channels: List[int]                # feature channels per UNet level
    attn_depths: List[bool]            # whether to use self-attention per level
    num_actions: int = 7               # 7-DoF robot action dimension
    is_upsampler: Optional[bool] = None  # set by Denoiser; keep False for base model


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()

        # Fourier features for the diffusion noise level sigma
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)

        # Action embedding: linear projection of 7-D actions over the conditioning window.
        # Each time step is embedded to (cond_channels // num_steps_conditioning) dims,
        # then all steps are flattened into a (cond_channels,)-D vector that can be
        # summed with the noise embedding.
        hidden_per_step = cfg.cond_channels // cfg.num_steps_conditioning
        self.act_emb = nn.Sequential(
            nn.Linear(cfg.num_actions, hidden_per_step),
            nn.ReLU(),
            nn.Flatten(),  # (B, T, hidden_per_step) -> (B, T * hidden_per_step) = (B, cond_channels)
        )

        # Conditioning projection: sums noise_emb + noise_cond_emb + act_emb, then
        # passes through a small MLP to produce the final conditioning vector used by
        # every AdaGroupNorm in the UNet.
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )

        # Input conv: num_steps_conditioning past fused frames + 1 noisy next frame,
        # each with img_channels (= 6) channels.
        # is_upsampler is always False for this model; the +1 term below matches the
        # original convention where the low-res guide frame occupies one extra slot.
        in_ch = (cfg.num_steps_conditioning + int(bool(cfg.is_upsampler)) + 1) * cfg.img_channels
        self.conv_in = Conv3x3(in_ch, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        # Output conv: predict all 6 channels (RGB + tactile residual)
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(
        self,
        noisy_next_obs: Tensor,   # (B, img_channels, H, W)  noisy target frame
        c_noise: Tensor,           # (B,)  log-sigma / 4 for the noise level
        c_noise_cond: Tensor,      # (B,)  log-sigma / 4 for the conditioning noise
        obs: Tensor,               # (B, n*img_channels, H, W)  concatenated past frames
        act: Optional[Tensor],     # (B, n, num_actions)  conditioning window actions
    ) -> Tensor:
        # Build conditioning embedding
        if act is not None:
            act_emb = self.act_emb(act)  # (B, cond_channels)
        else:
            act_emb = 0

        cond = self.cond_proj(
            self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb
        )  # (B, cond_channels)

        # Concatenate past frames and noisy next frame on the channel axis
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))  # (B, channels[0], H, W)

        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))  # (B, img_channels, H, W)
        return x
