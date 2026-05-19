"""
denoiser.py — Denoiser wrapper for the middle-fusion dual-stream world model.

This follows the EDM (Karras et al. 2022) preconditioning scheme:
  - c_in  : rescales the noisy input before feeding to the network
  - c_skip: weight for the skip connection (identity part)
  - c_out : weight for the network output
  - c_noise: log(sigma)/4, the noise-level embedding fed to the network

The denoiser wraps MiddleFusionInnerModel and provides:
  - forward()            : computes training loss (MSE on denoised output vs. target)
  - denoise()            : single denoising step for inference
  - compute_model_output(): raw network output (without skip connection)
  - wrap_model_output()  : applies skip + quantization
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.inner_model import MiddleFusionInnerModel, MiddleFusionInnerModelConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def add_dims(input: Tensor, n: int) -> Tensor:
    """Append (n - input.ndim) trailing size-1 dimensions."""
    return input.reshape(input.shape + (1,) * (n - input.ndim))


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class SigmaDistributionConfig:
    """Log-normal sigma distribution used during training."""
    loc: float = -1.2
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0


@dataclass
class DenoiserConfig:
    inner_model: MiddleFusionInnerModelConfig
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class Denoiser(nn.Module):
    """
    EDM-style denoiser for the middle-fusion dual-stream world model.

    Training:
      Reads batches with keys 'front' and 'tactile' (both (B, T, 3, H, W))
      and 'action' ((B, T, 7)).  For each autoregressive step i the model is
      asked to denoise the (n+i)-th front frame given n past front frames,
      n past tactile frames, and n past actions.

    Inference:
      Use denoise() directly or use DiffusionSampler.
    """

    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = MiddleFusionInnerModel(cfg.inner_model)
        self.sample_sigma_training: Optional[object] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.inner_model.parameters()).device

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None, "setup_training called twice"

        def sample_sigma(n: int, device: torch.device) -> Tensor:
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    # ------------------------------------------------------------------
    # EDM preconditioning helpers
    # ------------------------------------------------------------------

    def apply_noise(
        self, x: Tensor, sigma: Tensor, sigma_offset_noise: float
    ) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=x.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(
        self, sigma: Tensor, sigma_cond: Optional[Tensor]
    ) -> Conditioners:
        sigma = (sigma ** 2 + self.cfg.sigma_offset_noise ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.cfg.sigma_data ** 2).sqrt()
        c_skip = self.cfg.sigma_data ** 2 / (sigma ** 2 + self.cfg.sigma_data ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = (
            sigma_cond.log() / 4
            if sigma_cond is not None
            else torch.zeros_like(c_noise)
        )
        return Conditioners(
            *(
                add_dims(c, n)
                for c, n in zip(
                    (c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1)
                )
            )
        )

    def compute_model_output(
        self,
        noisy_next_rgb: Tensor,   # (B, 3, H, W)
        prev_rgb: Tensor,          # (B, n*3, H, W)
        prev_tactile: Tensor,      # (B, n*3, H, W)
        act: Tensor,               # (B, n, 7)
        cs: Conditioners,
    ) -> Tensor:
        rescaled_rgb = prev_rgb / self.cfg.sigma_data
        rescaled_noise = noisy_next_rgb * cs.c_in
        return self.inner_model(
            rescaled_noise,
            cs.c_noise,
            cs.c_noise_cond,
            rescaled_rgb,
            prev_tactile / self.cfg.sigma_data,
            act,
        )

    @torch.no_grad()
    def wrap_model_output(
        self, noisy_next_rgb: Tensor, model_output: Tensor, cs: Conditioners
    ) -> Tensor:
        d = cs.c_skip * noisy_next_rgb + cs.c_out * model_output
        # Quantise to {0, ..., 255} then map back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(
        self,
        noisy_next_rgb: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_rgb: Tensor,
        prev_tactile: Tensor,
        act: Tensor,
    ) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(
            noisy_next_rgb, prev_rgb, prev_tactile, act, cs
        )
        return self.wrap_model_output(noisy_next_rgb, model_output, cs)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict, device: torch.device):
        """
        Compute autoregressive training loss.

        batch keys:
          'front'   : (B, T, 3, H, W)  float32 in [-1, 1]
          'tactile' : (B, T, 3, H, W)  float32 in [-1, 1]
          'action'  : (B, T, 7)        float32 normalised

        For each autoregressive step i in [0, seq_length):
          prev_rgb     = all_front[:, i : n+i]          — n past RGB frames
          prev_tactile = all_tactile[:, i : n+i]        — n past tactile frames
          prev_act     = action[:, i : n+i]             — n past actions
          target       = all_front[:, n+i]              — next RGB frame to predict
        """
        front = batch["front"].to(device)     # (B, T, 3, H, W)
        tactile = batch["tactile"].to(device)  # (B, T, 3, H, W)
        act = batch["action"].to(device)       # (B, T, 7)

        b, t, c, h, w = front.shape
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n  # number of autoregressive prediction steps

        all_front = front.clone()
        # Tactile is used read-only (we only predict RGB)
        all_tactile = tactile

        loss = torch.tensor(0.0, device=device)

        for i in range(seq_length):
            # Build context windows
            prev_rgb = all_front[:, i : n + i].reshape(b, n * c, h, w)
            prev_tac = all_tactile[:, i : n + i].reshape(b, n * c, h, w)
            prev_act = act[:, i : n + i]           # (B, n, 7)
            target_rgb = all_front[:, n + i]        # (B, 3, H, W)

            # Optionally add noise to the conditioning frames
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, device)
                prev_rgb = self.apply_noise(prev_rgb, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            # Sample noise level and corrupt the target
            sigma = self.sample_sigma_training(b, device)
            noisy_rgb = self.apply_noise(target_rgb, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(
                noisy_rgb, prev_rgb, prev_tac, prev_act, cs
            )

            # EDM training target
            target = (target_rgb - cs.c_skip * noisy_rgb) / cs.c_out
            loss = loss + F.mse_loss(model_output, target)

            # Feed the denoised prediction back for the next autoregressive step
            denoised = self.wrap_model_output(noisy_rgb, model_output, cs)
            all_front[:, n + i] = denoised

        loss = loss / seq_length
        return loss, {"loss_denoising": loss.item()}
