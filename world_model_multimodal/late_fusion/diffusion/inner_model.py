from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int = 3           # always 3 — RGB output space
    cond_img_channels: int = 3      # channels of the conditioning modality (3 for RGB, 3 for tactile)
    num_steps_conditioning: int = 4
    cond_channels: int = 256
    action_dim: int = 7
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: List[int] = field(default_factory=lambda: [96, 96, 96, 96])
    attn_depths: List[int] = field(default_factory=lambda: [0, 0, 1, 1])
    is_upsampler: bool = False


class InnerModel(nn.Module):
    """Generic denoiser for one modality stream.

    conv_in takes: (num_steps_conditioning * cond_img_channels + img_channels) input channels:
      - Past conditioning frames concatenated: (num_steps_conditioning * cond_img_channels)
        These are prev_rgb frames for the RGB expert, or prev_tactile frames for the tactile expert.
      - Noisy next RGB frame: img_channels (= 3, always)

    conv_out produces: img_channels = 3 channels (predicted noise in RGB space).

    Both the RGB expert and tactile expert share this same class; the only difference is
    the value of cond_img_channels (which affects conv_in's input width) and what data
    is passed as the conditioning observation at call time.
    """

    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()

        self.cfg = cfg
        cond_channels = cfg.cond_channels
        num_steps = cfg.num_steps_conditioning
        action_dim = cfg.action_dim

        # Noise level embeddings — one for the target frame noise, one for the conditioning noise
        self.noise_emb = FourierFeatures(cond_channels)
        self.noise_cond_emb = FourierFeatures(cond_channels)

        # Action embedding: (B, num_steps, action_dim) → (B, cond_channels)
        # Each step gets cond_channels // num_steps dims, then all steps are concatenated by Flatten
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, cond_channels // num_steps),
            nn.ReLU(),
            nn.Flatten(),  # (B, num_steps, cond_channels // num_steps) -> (B, cond_channels)
        )

        # Projects the sum of noise + action embeddings into the conditioning vector
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        # conv_in: concatenate past conditioning frames + noisy next RGB frame
        # input channels = num_steps_conditioning * cond_img_channels  (past frames)
        #                + img_channels                                  (noisy target)
        conv_in_channels = cfg.num_steps_conditioning * cfg.cond_img_channels + cfg.img_channels
        self.conv_in = Conv3x3(conv_in_channels, cfg.channels[0])

        self.unet = UNet(cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        # Output always has img_channels = 3 (predicting noise in RGB space)
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(
        self,
        noisy_next_obs: Tensor,   # (B, img_channels, H, W)  — noisy next RGB frame
        c_noise: Tensor,          # (B,)                     — log-scaled noise level for target
        c_noise_cond: Tensor,     # (B,)                     — log-scaled noise level for conditioning
        obs: Tensor,              # (B, n * cond_img_channels, H, W) — past conditioning frames
        act: Tensor,              # (B, n, action_dim)        — past actions
    ) -> Tensor:
        """
        Returns the model's raw output (predicted "v" / noise direction) with shape
        (B, img_channels, H, W).
        """
        act_emb = self.act_emb(act)   # (B, cond_channels)
        cond = self.cond_proj(
            self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb
        )  # (B, cond_channels)

        # Concatenate past conditioning obs and the noisy target frame along channel dim
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
