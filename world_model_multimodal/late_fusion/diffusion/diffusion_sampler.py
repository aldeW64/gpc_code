"""
Late Fusion diffusion sampler.

Implements the same Euler / Heun stochastic sampler as the original DiffusionSampler
but calls LateFusionDenoiser.denoise_composed() so that both experts contribute to
each denoising step via additive score composition.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from diffusion.denoiser import LateFusionDenoiser


@dataclass
class LateFusionDiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 1          # 1 = Euler, 2 = Heun
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0
    s_cond: float = 0.0     # >0 → add noise to conditioning frames during sampling


def build_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: int,
    device: torch.device,
) -> Tensor:
    """Build the Karras et al. sigma schedule."""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


class LateFusionDiffusionSampler:
    """
    Euler / Heun stochastic sampler that composes scores from the RGB and tactile
    experts at every denoising step.

    Args:
        denoiser: A trained LateFusionDenoiser.
        cfg:      Sampler hyper-parameters.
    """

    def __init__(
        self,
        denoiser: LateFusionDenoiser,
        cfg: LateFusionDiffusionSamplerConfig,
    ) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(
            cfg.num_steps_denoising,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.rho,
            denoiser.device,
        )

    @torch.no_grad()
    def sample(
        self,
        prev_rgb: Tensor,          # (B, n, 3, H, W) — past RGB frames
        prev_tac: Tensor,          # (B, n, 3, H, W) — past tactile frames
        prev_act: Tensor,          # (B, n, action_dim) — past actions
        w_rgb: Optional[float] = None,
        w_tac: Optional[float] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Sample the next RGB frame by composing scores from both experts.

        Returns:
            x:          (B, 3, H, W) — sampled next RGB frame
            trajectory: list of intermediate denoised frames at each step
        """
        # Resolve composition weights (fall back to config defaults if not provided)
        if w_rgb is None:
            w_rgb = self.denoiser.composition_weights[0]
        if w_tac is None:
            w_tac = self.denoiser.composition_weights[1]

        device = prev_rgb.device
        b, n, c, h, w = prev_rgb.size()

        # Flatten time dimension: (B, n, 3, H, W) → (B, n*3, H, W)
        prev_rgb_flat = prev_rgb.reshape(b, n * c, h, w)
        prev_tac_flat = prev_tac.reshape(b, n * c, h, w)

        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)

        # Start from pure noise
        x = torch.randn(b, c, h, w, device=device) * self.sigmas[0]
        trajectory = [x]

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)

            # Stochastic churn: add extra noise
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # Optionally noise conditioning frames during sampling
            if self.cfg.s_cond > 0:
                sigma_cond = torch.full((b,), fill_value=self.cfg.s_cond, device=device)
                prev_rgb_cond = self.denoiser.apply_noise(prev_rgb_flat, sigma_cond, sigma_offset_noise=0)
                prev_tac_cond = self.denoiser.apply_noise(prev_tac_flat, sigma_cond, sigma_offset_noise=0)
            else:
                sigma_cond = None
                prev_rgb_cond = prev_rgb_flat
                prev_tac_cond = prev_tac_flat

            # Composed denoising step
            denoised = self.denoiser.denoise_composed(
                noisy_next_rgb=x,
                sigma=sigma_hat * s_in,
                sigma_cond=sigma_cond,
                prev_rgb=prev_rgb_cond,
                prev_tac=prev_tac_cond,
                act=prev_act,
                w_rgb=w_rgb,
                w_tac=w_tac,
            )

            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                # Euler step
                x = x + d * dt
            else:
                # Heun's second-order correction
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise_composed(
                    noisy_next_rgb=x_2,
                    sigma=next_sigma * s_in,
                    sigma_cond=sigma_cond,
                    prev_rgb=prev_rgb_cond,
                    prev_tac=prev_tac_cond,
                    act=prev_act,
                    w_rgb=w_rgb,
                    w_tac=w_tac,
                )
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt

            trajectory.append(x)

        return x, trajectory
