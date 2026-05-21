"""
diffusion_sampler.py — Stochastic sampler (Karras et al. 2022) for the
middle-fusion dual-stream denoiser.

Both RGB and tactile are prediction targets, so the sampler maintains two
parallel ODE trajectories (x_rgb, x_tac) that are denoised jointly at every
step via the dual-stream inner model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from diffusion.denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 1          # 1 = Euler, 2 = Heun
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0
    s_cond: float = 0.0     # if > 0, add this noise level to the conditioning frames


def build_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: int,
    device: torch.device,
) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class DiffusionSampler:
    """
    Karras-style ODE sampler for the middle-fusion denoiser.

    Maintains two parallel noise trajectories (x_rgb, x_tac) that are
    denoised jointly. Returns both predicted next frames.
    """

    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
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
        prev_rgb: Tensor,      # (B, n, 3, H, W)  past RGB frames
        prev_tactile: Tensor,  # (B, n, 3, H, W)  past tactile frames
        prev_act: Tensor,      # (B, n, 7)         past actions
    ) -> Tuple[Tuple[Tensor, Tensor], List[Tensor]]:
        """
        Jointly denoise from pure Gaussian noise to predicted next RGB and tactile frames.

        Returns:
            (x_rgb, x_tac) : each (B, 3, H, W) — denoised predictions
            trajectory     : list of intermediate x_rgb tensors (one per step)
        """
        device = prev_rgb.device
        b, n, c, h, w = prev_rgb.shape

        prev_rgb_flat = prev_rgb.reshape(b, n * c, h, w)
        prev_tac_flat = prev_tactile.reshape(b, n * c, h, w)

        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)

        # Start both trajectories from independent Gaussian noise
        x_rgb = torch.randn(b, c, h, w, device=device)
        x_tac = torch.randn(b, c, h, w, device=device)
        trajectory: List[Tensor] = [x_rgb]

        prev_rgb_cond = prev_rgb_flat
        prev_tac_cond = prev_tac_flat

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0.0
            sigma_hat = sigma * (1 + gamma)

            if gamma > 0:
                x_rgb = x_rgb + torch.randn_like(x_rgb) * self.cfg.s_noise * (sigma_hat ** 2 - sigma ** 2) ** 0.5
                x_tac = x_tac + torch.randn_like(x_tac) * self.cfg.s_noise * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            if self.cfg.s_cond > 0:
                sigma_cond = torch.full((b,), self.cfg.s_cond, device=device)
                prev_rgb_cond = self.denoiser.apply_noise(prev_rgb_flat, sigma_cond, 0.0)
                prev_tac_cond = self.denoiser.apply_noise(prev_tac_flat, sigma_cond, 0.0)
                sigma_cond_arg: Optional[Tensor] = sigma_cond
            else:
                sigma_cond_arg = None

            denoised_rgb, denoised_tac = self.denoiser.denoise(
                x_rgb, x_tac,
                sigma_hat * s_in,
                sigma_cond_arg,
                prev_rgb_cond,
                prev_tac_cond,
                prev_act,
            )

            d_rgb = (x_rgb - denoised_rgb) / sigma_hat
            d_tac = (x_tac - denoised_tac) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                x_rgb = x_rgb + d_rgb * dt
                x_tac = x_tac + d_tac * dt
            else:
                # Heun's second-order correction
                x_rgb_2 = x_rgb + d_rgb * dt
                x_tac_2 = x_tac + d_tac * dt
                denoised_rgb_2, denoised_tac_2 = self.denoiser.denoise(
                    x_rgb_2, x_tac_2,
                    next_sigma * s_in,
                    sigma_cond_arg,
                    prev_rgb_cond,
                    prev_tac_cond,
                    prev_act,
                )
                d_rgb_2 = (x_rgb_2 - denoised_rgb_2) / next_sigma
                d_tac_2 = (x_tac_2 - denoised_tac_2) / next_sigma
                x_rgb = x_rgb + (d_rgb + d_rgb_2) / 2 * dt
                x_tac = x_tac + (d_tac + d_tac_2) / 2 * dt

            trajectory.append(x_rgb)

        return (x_rgb, x_tac), trajectory
