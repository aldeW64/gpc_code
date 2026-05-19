"""
Late Fusion (Score Composition) denoiser.

Two independent InnerModel experts are trained jointly:
  - model_rgb: conditioned on prev_rgb + actions → predicts eps for next RGB frame
  - model_tac: conditioned on prev_tactile + actions → also predicts eps for the same RGB frame

At inference, their score estimates are composed:
    eps_composed = w_rgb * eps_rgb + w_tac * eps_tac
which in the EDM / preconditioning framework translates to composing model outputs
before the c_skip / c_out wrapping step.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.inner_model import InnerModel, InnerModelConfig


def add_dims(input: Tensor, n: int) -> Tensor:
    """Expand a 1-D tensor to n dimensions by appending singleton dims."""
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor      # input scaling  (B,1,1,1)
    c_out: Tensor     # output scaling (B,1,1,1)
    c_skip: Tensor    # skip scaling   (B,1,1,1)
    c_noise: Tensor   # noise emb input (B,)
    c_noise_cond: Tensor  # cond noise emb input (B,)


@dataclass
class SigmaDistributionConfig:
    loc: float = -1.2
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0


@dataclass
class LateFusionDenoiserConfig:
    # Shared architecture settings for both experts
    img_channels: int = 3              # output/target space (RGB)
    num_steps_conditioning: int = 4    # number of past frames fed as conditioning
    cond_channels: int = 256
    action_dim: int = 7
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: List[int] = field(default_factory=lambda: [96, 96, 96, 96])
    attn_depths: List[int] = field(default_factory=lambda: [0, 0, 1, 1])

    # EDM preconditioning
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True

    # Training loss balance: loss = alpha * loss_rgb + (1-alpha) * loss_tac
    loss_alpha: float = 0.5

    # Inference composition weights [w_rgb, w_tac]
    composition_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])


class LateFusionDenoiser(nn.Module):
    """
    Late-fusion world model with two denoiser experts:
      - model_rgb: denoises next RGB frame conditioned on past RGB frames + actions
      - model_tac: denoises next RGB frame conditioned on past tactile frames + actions

    Both predict in the same output space (RGB). During sampling, their outputs are
    composed via weighted sum before the skip-connection wrapping step, implementing
    additive score composition in the EDM framework.
    """

    def __init__(self, cfg: LateFusionDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sigma_data = cfg.sigma_data

        # RGB expert: conditioning modality is RGB (cond_img_channels = 3)
        cfg_rgb = InnerModelConfig(
            img_channels=cfg.img_channels,
            cond_img_channels=3,
            num_steps_conditioning=cfg.num_steps_conditioning,
            cond_channels=cfg.cond_channels,
            action_dim=cfg.action_dim,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
            is_upsampler=False,
        )

        # Tactile expert: conditioning modality is tactile (cond_img_channels = 3)
        # Same channel count since both modalities are 3-channel images after resize
        cfg_tac = InnerModelConfig(
            img_channels=cfg.img_channels,
            cond_img_channels=3,
            num_steps_conditioning=cfg.num_steps_conditioning,
            cond_channels=cfg.cond_channels,
            action_dim=cfg.action_dim,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
            is_upsampler=False,
        )

        self.model_rgb = InnerModel(cfg_rgb)
        self.model_tac = InnerModel(cfg_tac)

        self.alpha = cfg.loss_alpha
        self.composition_weights = cfg.composition_weights

        # Will be set by setup_training()
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.model_rgb.noise_emb.weight.device

    # ------------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------------

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        """Register the log-normal sigma sampler used during training."""
        assert self.sample_sigma_training is None, "setup_training called twice"

        def sample_sigma(n: int, device: torch.device) -> Tensor:
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    # ------------------------------------------------------------------
    # EDM preconditioning (shared by both experts)
    # ------------------------------------------------------------------

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        """Add offset noise + isotropic Gaussian noise scaled by sigma."""
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        """Compute EDM preconditioning scalars from sigma values."""
        sigma = (sigma ** 2 + self.cfg.sigma_offset_noise ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = (
            sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        )
        return Conditioners(
            *(add_dims(c, n) for c, n in zip(
                (c_in, c_out, c_skip, c_noise, c_noise_cond),
                (4, 4, 4, 1, 1),
            ))
        )

    # ------------------------------------------------------------------
    # Per-expert model calls
    # ------------------------------------------------------------------

    def _call_rgb_expert(
        self,
        noisy_next_rgb: Tensor,
        prev_rgb: Tensor,
        act: Tensor,
        cs: Conditioners,
    ) -> Tensor:
        """Raw model output from the RGB expert (before c_skip / c_out wrapping)."""
        return self.model_rgb(
            noisy_next_rgb * cs.c_in,
            cs.c_noise,
            cs.c_noise_cond,
            prev_rgb / self.sigma_data,
            act,
        )

    def _call_tac_expert(
        self,
        noisy_next_rgb: Tensor,
        prev_tac: Tensor,
        act: Tensor,
        cs: Conditioners,
    ) -> Tensor:
        """Raw model output from the tactile expert (before c_skip / c_out wrapping)."""
        return self.model_tac(
            noisy_next_rgb * cs.c_in,
            cs.c_noise,
            cs.c_noise_cond,
            prev_tac / self.sigma_data,
            act,
        )

    # ------------------------------------------------------------------
    # Output wrapping (quantize denoised estimate back to [-1, 1] grid)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def wrap_model_output(
        self,
        noisy_next_obs: Tensor,
        model_output: Tensor,
        cs: Conditioners,
    ) -> Tensor:
        """
        Apply EDM skip connection and quantize to {0,...,255}/255*2-1 pixel grid.
        d = c_skip * x_noisy + c_out * model_output, then quantize.
        """
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    # ------------------------------------------------------------------
    # Single-step denoise (for sampling / evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def denoise_rgb(
        self,
        noisy_next_rgb: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_rgb: Tensor,
        act: Tensor,
    ) -> Tensor:
        """Denoise using only the RGB expert."""
        cs = self.compute_conditioners(sigma, sigma_cond)
        out = self._call_rgb_expert(noisy_next_rgb, prev_rgb, act, cs)
        return self.wrap_model_output(noisy_next_rgb, out, cs)

    @torch.no_grad()
    def denoise_tac(
        self,
        noisy_next_rgb: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_tac: Tensor,
        act: Tensor,
    ) -> Tensor:
        """Denoise using only the tactile expert."""
        cs = self.compute_conditioners(sigma, sigma_cond)
        out = self._call_tac_expert(noisy_next_rgb, prev_tac, act, cs)
        return self.wrap_model_output(noisy_next_rgb, out, cs)

    @torch.no_grad()
    def denoise_composed(
        self,
        noisy_next_rgb: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_rgb: Tensor,
        prev_tac: Tensor,
        act: Tensor,
        w_rgb: float = 0.7,
        w_tac: float = 0.3,
    ) -> Tensor:
        """
        Compose score estimates from both experts.

        In the EDM epsilon-prediction parameterisation, additive score composition
        corresponds to a weighted sum of the raw model outputs (before the
        c_skip / c_out wrapping):

            composed_output = w_rgb * out_rgb + w_tac * out_tac

        The skip-connection wrapping is then applied once on top of the composed output.
        This faithfully implements
            ∇_x log p(x | rgb, tac) ≈ w_rgb * score_rgb + w_tac * score_tac
        without double-counting the prior.
        """
        cs = self.compute_conditioners(sigma, sigma_cond)
        out_rgb = self._call_rgb_expert(noisy_next_rgb, prev_rgb, act, cs)
        out_tac = self._call_tac_expert(noisy_next_rgb, prev_tac, act, cs)
        composed_output = w_rgb * out_rgb + w_tac * out_tac
        return self.wrap_model_output(noisy_next_rgb, composed_output, cs)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Autoregressive multi-step training forward pass.

        batch keys:
          'front'   : (B, T, 3, H, W)  — RGB frames, normalised to [-1, 1]
          'tactile' : (B, T, 3, H, W)  — tactile frames, normalised to [-1, 1]
          'action'  : (B, T, 7)         — actions, normalised to [-1, 1]

        T = obs_horizon + pred_horizon (= num_steps_conditioning + seq_length)

        Autoregressive loop:
          - For each future step i:
              * Both experts share the same noisy target (noisy next RGB frame)
              * RGB expert: conditioned on rolling prev_rgb (initially GT, then model prediction)
              * Tactile expert: conditioned on GT tactile (never predicted autoregressively)
              * Loss = alpha * MSE(rgb_out, target) + (1-alpha) * MSE(tac_out, target)
              * After each step, update prev_rgb with the RGB expert's denoised output
                (teacher-forced rollout for training stability, following phase-two practice)
        """
        front = batch['front'].to(device)     # (B, T, 3, H, W)
        tactile = batch['tactile'].to(device)  # (B, T, 3, H, W)
        act = batch['action'].to(device)       # (B, T, 7)

        b, t, c, h, w = front.size()
        n = self.cfg.num_steps_conditioning
        seq_length = t - n   # number of autoregressive prediction steps

        # Working copy of RGB frames — gets overwritten with model predictions
        # after each step (teacher-forcing on its own predictions, matching phase-two).
        all_rgb = front.clone()   # (B, T, 3, H, W)

        total_loss = torch.tensor(0.0, device=device)
        total_loss_rgb = 0.0
        total_loss_tac = 0.0

        for i in range(seq_length):
            # ---- Conditioning frames ----
            # RGB expert sees rolling predictions for prev frames
            prev_rgb = all_rgb[:, i: n + i].reshape(b, n * c, h, w)      # (B, n*3, H, W)
            # Tactile expert always sees ground-truth tactile frames
            prev_tac = tactile[:, i: n + i].reshape(b, n * c, h, w)      # (B, n*3, H, W)
            # Actions aligned to the conditioning window
            prev_act = act[:, i: n + i]                                    # (B, n, 7)
            # Ground-truth next RGB frame (target)
            target_rgb = all_rgb[:, n + i]                                 # (B, 3, H, W)

            # ---- Optional conditioning noise ----
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, device)
                prev_rgb_noised = self.apply_noise(prev_rgb, sigma_cond, self.cfg.sigma_offset_noise)
                prev_tac_noised = self.apply_noise(prev_tac, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None
                prev_rgb_noised = prev_rgb
                prev_tac_noised = prev_tac

            # ---- Sample noise for the target frame ----
            sigma = self.sample_sigma_training(b, device)
            noisy_obs = self.apply_noise(target_rgb, sigma, self.cfg.sigma_offset_noise)

            # ---- Forward through both experts ----
            cs = self.compute_conditioners(sigma, sigma_cond)

            out_rgb = self._call_rgb_expert(noisy_obs, prev_rgb_noised, prev_act, cs)
            out_tac = self._call_tac_expert(noisy_obs, prev_tac_noised, prev_act, cs)

            # ---- Compute target in denoiser output space ----
            # target = (x_clean - c_skip * x_noisy) / c_out
            target = (target_rgb - cs.c_skip * noisy_obs) / cs.c_out

            # ---- Per-expert losses ----
            loss_rgb = F.mse_loss(out_rgb, target)
            loss_tac = F.mse_loss(out_tac, target)
            step_loss = self.alpha * loss_rgb + (1.0 - self.alpha) * loss_tac
            total_loss = total_loss + step_loss
            total_loss_rgb += loss_rgb.item()
            total_loss_tac += loss_tac.item()

            # ---- Update rolling RGB buffer with RGB expert's denoised output ----
            # This implements the autoregressive teacher-forcing strategy from phase-two:
            # the model's own RGB predictions become the conditioning for future steps.
            # Tactile is never predicted — always read from GT.
            with torch.no_grad():
                denoised = self.wrap_model_output(noisy_obs, out_rgb, cs)
            all_rgb[:, n + i] = denoised.detach()

        total_loss = total_loss / seq_length
        metrics = {
            "loss": total_loss.item(),
            "loss_rgb": total_loss_rgb / seq_length,
            "loss_tac": total_loss_tac / seq_length,
        }
        return total_loss, metrics
