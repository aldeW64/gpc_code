"""
Early Fusion Denoiser
=====================
Adapts the original denoiser to the multimodal (RGB + tactile) setting.

Changes from the PushT denoiser:
  - batch keys are 'front' and 'tactile' (each B,T,3,H,W) instead of 'image'
  - they are concatenated along the channel dim -> (B,T,6,H,W) before processing
  - actions are 7-D robot actions instead of 2-D pixel actions
  - img_channels is 6 throughout (conv_in, conv_out, loss)
  - no upsampler path (kept for completeness, always False here)
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.inner_model import InnerModel, InnerModelConfig


def add_dims(input: Tensor, n: int) -> Tensor:
    """Expand a 1-D tensor to n dimensions by appending singleton axes."""
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    upsampling_factor: Optional[int] = None


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        cfg.inner_model.is_upsampler = self.is_upsampler
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device) -> Tensor:
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip(
            (c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1)
        )))

    def compute_model_output(
        self,
        noisy_next_obs: Tensor,
        obs: Tensor,
        act: Optional[Tensor],
        cs: Conditioners,
    ) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)

    @torch.no_grad()
    def wrap_model_output(
        self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners
    ) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0,...,255} and back to [-1,1] (matches original convention)
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(
        self,
        noisy_next_obs: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        obs: Tensor,
        act: Optional[Tensor],
    ) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        return self.wrap_model_output(noisy_next_obs, model_output, cs)

    def forward(self, batch: dict, device: torch.device):
        """
        Training forward pass.

        batch keys expected:
            'front'   : (B, T, 3, H, W) float32 in [-1, 1]
            'tactile' : (B, T, 3, H, W) float32 in [-1, 1]
            'action'  : (B, T, 7) float32 in [-1, 1]

        Returns (loss, metrics_dict).
        """
        front = batch['front'].to(device)    # (B, T, 3, H, W)
        tactile = batch['tactile'].to(device)  # (B, T, 3, H, W)
        act = batch['action'].to(device)      # (B, T, 7)

        # Early fusion: concatenate RGB and tactile along the channel dimension
        obs = torch.cat([front, tactile], dim=2)  # (B, T, 6, H, W)

        b, t, c, h, w = obs.size()
        # c should be 6 = img_channels
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n  # number of autoregressive prediction steps

        all_obs = obs.clone()

        loss = 0.0
        for i in range(seq_length):
            # Past n frames, flattened on the channel axis: (B, n*c, H, W)
            prev_obs = all_obs[:, i: n + i].reshape(b, n * c, h, w)
            # Actions for the conditioning window: (B, n, 7)
            prev_act = act[:, i: n + i]
            # Ground-truth next frame: (B, c, H, W)
            next_obs = all_obs[:, n + i]

            # Optionally add noise to the conditioning frames
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            # Sample noise level for the target frame
            sigma = self.sample_sigma_training(b, device)
            noisy_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)
            model_output = self.compute_model_output(noisy_obs, prev_obs, prev_act, cs)

            # EDM-style target
            target = (next_obs - cs.c_skip * noisy_obs) / cs.c_out
            loss += F.mse_loss(model_output, target)

            # Update all_obs with the denoised prediction so subsequent steps
            # condition on the model's own outputs (autoregressive training)
            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss = loss / seq_length
        return loss, {"loss_denoising": loss.item()}
