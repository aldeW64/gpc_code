"""
denoiser.py — Denoiser wrapper for the middle-fusion dual-stream world model.

Follows the EDM (Karras et al. 2022) preconditioning scheme. Both RGB and
tactile are now prediction targets: each autoregressive step denoises a next
RGB frame AND a next tactile frame jointly via the dual-stream inner model.
The training loss is the sum of MSE over both modalities.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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
      Reads batches with keys 'front' (B,T,3,H,W), 'tactile' (B,T,3,H,W),
      and 'action' (B,T,7). For each autoregressive step the model jointly
      denoises the next RGB frame and the next tactile frame, conditioned on
      n past frames of both modalities plus n past actions.

    Inference:
      Use denoise() directly or use DiffusionSampler.
    """

    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = MiddleFusionInnerModel(cfg.inner_model)
        self.sample_sigma_training: Optional[object] = None

    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.inner_model.parameters()).device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None, "setup_training called twice"

        def sample_sigma(n: int, device: torch.device) -> Tensor:
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    # ------------------------------------------------------------------
    # EDM preconditioning helpers
    # ------------------------------------------------------------------

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
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
        noisy_next_tac: Tensor,   # (B, 3, H, W)
        prev_rgb: Tensor,          # (B, n*3, H, W)
        prev_tactile: Tensor,      # (B, n*3, H, W)
        act: Tensor,               # (B, n, 7)
        cs: Conditioners,
    ) -> Tuple[Tensor, Tensor]:
        return self.inner_model(
            noisy_next_rgb * cs.c_in,
            noisy_next_tac * cs.c_in,
            cs.c_noise,
            cs.c_noise_cond,
            prev_rgb / self.cfg.sigma_data,
            prev_tactile / self.cfg.sigma_data,
            act,
        )

    @torch.no_grad()
    def wrap_model_output(
        self, noisy: Tensor, model_output: Tensor, cs: Conditioners
    ) -> Tensor:
        d = cs.c_skip * noisy + cs.c_out * model_output
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(
        self,
        noisy_next_rgb: Tensor,
        noisy_next_tac: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_rgb: Tensor,
        prev_tactile: Tensor,
        act: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        cs = self.compute_conditioners(sigma, sigma_cond)
        out_rgb, out_tac = self.compute_model_output(
            noisy_next_rgb, noisy_next_tac, prev_rgb, prev_tactile, act, cs
        )
        return (
            self.wrap_model_output(noisy_next_rgb, out_rgb, cs),
            self.wrap_model_output(noisy_next_tac, out_tac, cs),
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict, device: torch.device):
        """
        Autoregressive training loss over both RGB and tactile modalities.

        batch keys:
          'front'   : (B, T, 3, H, W)  float32 in [-1, 1]
          'tactile' : (B, T, 3, H, W)  float32 in [-1, 1]
          'action'  : (B, T, 7)        float32 normalised

        For each step i in [0, seq_length):
          Denoises front[:, n+i] and tactile[:, n+i] jointly, conditioned on
          past n frames of both modalities. Both buffers are updated with the
          model's own predictions for subsequent steps.
        """
        front = batch["front"].to(device)     # (B, T, 3, H, W)
        tactile = batch["tactile"].to(device)  # (B, T, 3, H, W)
        act = batch["action"].to(device)       # (B, T, 7)

        b, t, c, h, w = front.shape
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = t - n

        all_front = front.clone()
        all_tactile = tactile.clone()  # both buffers roll out autoregressively

        loss = torch.tensor(0.0, device=device)

        for i in range(seq_length):
            prev_rgb = all_front[:, i : n + i].reshape(b, n * c, h, w)
            prev_tac = all_tactile[:, i : n + i].reshape(b, n * c, h, w)
            prev_act = act[:, i : n + i]
            target_rgb = all_front[:, n + i]
            target_tac = all_tactile[:, n + i]

            # Noise both conditioning streams consistently
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, device)
                prev_rgb = self.apply_noise(prev_rgb, sigma_cond, self.cfg.sigma_offset_noise)
                prev_tac = self.apply_noise(prev_tac, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            # Sample one noise level and corrupt both targets
            sigma = self.sample_sigma_training(b, device)
            noisy_rgb = self.apply_noise(target_rgb, sigma, self.cfg.sigma_offset_noise)
            noisy_tac = self.apply_noise(target_tac, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)
            out_rgb, out_tac = self.compute_model_output(
                noisy_rgb, noisy_tac, prev_rgb, prev_tac, prev_act, cs
            )

            target_rgb_edm = (target_rgb - cs.c_skip * noisy_rgb) / cs.c_out
            target_tac_edm = (target_tac - cs.c_skip * noisy_tac) / cs.c_out
            loss = loss + F.mse_loss(out_rgb, target_rgb_edm) + F.mse_loss(out_tac, target_tac_edm)

            with torch.no_grad():
                denoised_rgb = self.wrap_model_output(noisy_rgb, out_rgb, cs)
                denoised_tac = self.wrap_model_output(noisy_tac, out_tac, cs)
            all_front[:, n + i] = denoised_rgb.detach()
            all_tactile[:, n + i] = denoised_tac.detach()

        loss = loss / seq_length
        return loss, {"loss_denoising": loss.item()}
