"""
inner_model.py — Dual-stream inner model for middle-fusion multimodal world model.

Architecture:
  RGB stream   : (prev_rgb  concatenated with noisy_next_rgb) → conv_in_rgb → RGB encoder
  Tactile stream: prev_tactile → conv_in_tac → tactile encoder
  Bottleneck   : CrossModalAttention fuses both streams
  Decoder      : shared decoder (using RGB skip connections) → conv_out → predicted next RGB frame

Conditioning:
  - Diffusion noise level sigma via Fourier features
  - Previous-observation noise level via Fourier features
  - 7-DoF robot actions via a per-step linear embedding, flattened to cond_channels
  All three are summed and passed through a small MLP to give the global FiLM conditioning
  vector consumed by every AdaGroupNorm in the two encoder streams and the decoder.
"""

from dataclasses import dataclass, field
from typing import List, Optional

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
      RGB stream  : (num_steps_conditioning + 1) * img_channels
                    = 4 past frames + 1 noisy next frame = 5 × 3 = 15 channels
      Tactile stream: num_steps_conditioning * img_channels
                    = 4 past frames = 4 × 3 = 12 channels

    Output: predicted noise residual for the next RGB frame, shape (B, 3, H, W).
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
        # RGB: n past frames + 1 noisy next frame
        rgb_in_ch = (n + 1) * cfg.img_channels
        self.conv_in_rgb = Conv3x3(rgb_in_ch, cfg.channels[0])

        # Tactile: n past frames only (no noisy future tactile)
        tac_in_ch = n * cfg.img_channels
        self.conv_in_tac = Conv3x3(tac_in_ch, cfg.channels[0])

        # ---- Dual-stream UNet (encoders + cross-modal attn + decoder) ------
        self.dual_unet = DualStreamUNet(
            cond_channels=cfg.cond_channels,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
        )

        # ---- Output head ---------------------------------------------------
        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_next_rgb: Tensor,   # (B, 3, H, W)         noisy target RGB frame
        c_noise: Tensor,           # (B,)                  log-sigma / 4
        c_noise_cond: Tensor,      # (B,)                  log-sigma_cond / 4
        prev_rgb: Tensor,          # (B, n*3, H, W)        concatenated past RGB frames
        prev_tactile: Tensor,      # (B, n*3, H, W)        concatenated past tactile frames
        act: Tensor,               # (B, n, num_actions)   conditioning window actions
    ) -> Tensor:
        # ---- Build global conditioning vector ------------------------------
        act_emb = self.act_emb(act)                    # (B, cond_channels)
        cond = self.cond_proj(
            self.noise_emb(c_noise)
            + self.noise_cond_emb(c_noise_cond)
            + act_emb
        )                                              # (B, cond_channels)

        # ---- Project raw pixels into feature space -------------------------
        # RGB stream: past frames + noisy next frame concatenated on channel dim
        rgb_input = torch.cat([prev_rgb, noisy_next_rgb], dim=1)  # (B, (n+1)*3, H, W)
        rgb_feat = self.conv_in_rgb(rgb_input)                    # (B, channels[0], H, W)

        # Tactile stream: past frames only
        tac_feat = self.conv_in_tac(prev_tactile)                 # (B, channels[0], H, W)

        # ---- Dual-stream forward (encode → cross-modal → decode) -----------
        out = self.dual_unet(rgb_feat, tac_feat, cond)            # (B, channels[0], H, W)

        # ---- Output projection ---------------------------------------------
        out = self.conv_out(F.silu(self.norm_out(out)))           # (B, 3, H, W)
        return out
