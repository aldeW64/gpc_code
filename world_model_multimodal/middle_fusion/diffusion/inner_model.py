"""
inner_model.py — Dual-stream inner model for middle-fusion multimodal world model.

Architecture:
  RGB stream   : (prev_rgb | noisy_next_rgb) → conv_in_rgb → RGB encoder
  Tactile stream: (prev_tactile | noisy_next_tac) → conv_in_tac → tactile encoder
  Bottleneck   : CrossModalAttention fuses both streams bidirectionally
  RGB decoder  : fused RGB bottleneck + RGB skip connections → conv_out_rgb → next RGB
  Tactile decoder: fused tactile bottleneck + tactile skip connections → conv_out_tac → next tactile

Both modalities are prediction targets; each stream receives its own noisy next frame as input.

Conditioning:
  - Diffusion noise level sigma via Fourier features
  - Previous-observation noise level via Fourier features
  - 7-DoF robot actions via a per-step linear embedding, flattened to cond_channels
  All three are summed and passed through a small MLP to give the global FiLM conditioning
  vector consumed by every AdaGroupNorm in the two encoder streams and the two decoders.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.blocks import Conv3x3, DualStreamUNet, FourierFeatures, GroupNorm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MiddleFusionInnerModelConfig:
    """Configuration for MiddleFusionInnerModel."""

    # ---- image dimensions ---------------------------------------------------
    img_channels: int = 3             # RGB channels per frame
    num_steps_conditioning: int = 4   # number of past frames used as context (n)

    # ---- action embedding ---------------------------------------------------
    num_actions: int = 7              # 7-DoF robot action dimension

    # ---- UNet / conditioning -----------------------------------------------
    cond_channels: int = 256          # dimension of the global conditioning vector
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: List[int] = field(default_factory=lambda: [96, 96, 96, 96])
    attn_depths: List[int] = field(default_factory=lambda: [0, 0, 1, 1])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MiddleFusionInnerModel(nn.Module):
    """
    Dual-stream diffusion denoiser with cross-modal attention at the bottleneck.

    Input channels:
      RGB stream    : (num_steps_conditioning + 1) * img_channels = 5 × 3 = 15 channels
                      (4 past RGB frames + 1 noisy next RGB frame)
      Tactile stream: (num_steps_conditioning + 1) * img_channels = 5 × 3 = 15 channels
                      (4 past tactile frames + 1 noisy next tactile frame)

    Output: tuple of predicted noise residuals —
      out_rgb: (B, 3, H, W)  — noise residual for the next RGB frame
      out_tac: (B, 3, H, W)  — noise residual for the next tactile frame
    """

    def __init__(self, cfg: MiddleFusionInnerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = cfg.num_steps_conditioning

        # ---- Noise-level embeddings ----------------------------------------
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)

        # ---- Action embedding ----------------------------------------------
        # Each time step maps num_actions → (cond_channels // n) dimensions.
        # After the linear, the (B, n, hidden_per_step) tensor is flattened to
        # (B, cond_channels) so it can be added to the noise embeddings.
        hidden_per_step = cfg.cond_channels // n
        self.act_emb = nn.Sequential(
            nn.Linear(cfg.num_actions, hidden_per_step),
            nn.ReLU(),
            nn.Flatten(),  # (B, n, hidden_per_step) → (B, n * hidden_per_step) = (B, cond_channels)
        )

        # ---- Conditioning projection ----------------------------------------
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )

        # ---- Input conv layers (one per stream) ----------------------------
        # Both streams: n past frames + 1 noisy next frame for that modality
        in_ch = (n + 1) * cfg.img_channels
        self.conv_in_rgb = Conv3x3(in_ch, cfg.channels[0])
        self.conv_in_tac = Conv3x3(in_ch, cfg.channels[0])

        # ---- Dual-stream UNet (encoders + cross-modal attn + two decoders) --
        self.dual_unet = DualStreamUNet(
            cond_channels=cfg.cond_channels,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
        )

        # ---- Separate output heads (one per modality) ----------------------
        self.norm_out_rgb = GroupNorm(cfg.channels[0])
        self.conv_out_rgb = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out_rgb.weight)

        self.norm_out_tac = GroupNorm(cfg.channels[0])
        self.conv_out_tac = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out_tac.weight)

    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_next_rgb: Tensor,   # (B, 3, H, W)       noisy target RGB frame
        noisy_next_tac: Tensor,   # (B, 3, H, W)       noisy target tactile frame
        c_noise: Tensor,           # (B,)               log-sigma / 4
        c_noise_cond: Tensor,      # (B,)               log-sigma_cond / 4
        prev_rgb: Tensor,          # (B, n*3, H, W)     concatenated past RGB frames
        prev_tactile: Tensor,      # (B, n*3, H, W)     concatenated past tactile frames
        act: Tensor,               # (B, n, num_actions) conditioning window actions
    ) -> Tuple[Tensor, Tensor]:
        # ---- Build global conditioning vector ------------------------------
        act_emb = self.act_emb(act)
        cond = self.cond_proj(
            self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb
        )  # (B, cond_channels)

        # ---- Project raw pixels into feature space -------------------------
        # Each stream receives its own past frames + noisy next target frame
        rgb_feat = self.conv_in_rgb(
            torch.cat([prev_rgb, noisy_next_rgb], dim=1)     # (B, (n+1)*3, H, W)
        )
        tac_feat = self.conv_in_tac(
            torch.cat([prev_tactile, noisy_next_tac], dim=1)  # (B, (n+1)*3, H, W)
        )

        # ---- Dual-stream forward (encode → cross-modal → two decoders) ----
        rgb_out_feat, tac_out_feat = self.dual_unet(rgb_feat, tac_feat, cond)

        # ---- Separate output heads -----------------------------------------
        out_rgb = self.conv_out_rgb(F.silu(self.norm_out_rgb(rgb_out_feat)))  # (B, 3, H, W)
        out_tac = self.conv_out_tac(F.silu(self.norm_out_tac(tac_out_feat)))  # (B, 3, H, W)
        return out_rgb, out_tac
