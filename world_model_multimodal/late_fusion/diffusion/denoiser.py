"""
Late Fusion (Score Composition) denoiser.

Two independent InnerModel experts are trained jointly:
  - model_rgb: conditioned on prev_rgb  → predicts noise for the next RGB frame
  - model_tac: conditioned on prev_tac  → predicts noise for the next tactile frame

A learned WeightPredictor blends the two expert outputs to produce predictions
for both modalities:

    composed_rgb = w_for_rgb[0] * out_rgb + w_for_rgb[1] * out_tac
    composed_tac = w_for_tac[0] * out_rgb + w_for_tac[1] * out_tac

where weights are softmax-normalized pairs predicted from the conditioning vector
(sigma, sigma_cond, actions). Both RGB and tactile are autoregressive prediction
targets; both rolling buffers are updated each step.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusion.inner_model import InnerModel, InnerModelConfig, WeightPredictor
from diffusion.blocks import FourierFeatures


def add_dims(input: Tensor, n: int) -> Tensor:
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
    loc: float = -1.2
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0


@dataclass
class LateFusionDenoiserConfig:
    img_channels: int = 3
    num_steps_conditioning: int = 4
    cond_channels: int = 256
    action_dim: int = 7
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: List[int] = field(default_factory=lambda: [96, 96, 96, 96])
    attn_depths: List[int] = field(default_factory=lambda: [0, 0, 1, 1])

    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True

    # Training loss balance: loss = alpha * loss_rgb + (1-alpha) * loss_tac
    loss_alpha: float = 0.5


class LateFusionDenoiser(nn.Module):
    """
    Late-fusion world model with two expert denoisers and a learned weight predictor.

    model_rgb: denoises next RGB frame conditioned on past RGB frames + actions
    model_tac: denoises next tactile frame conditioned on past tactile frames + actions

    At each step, the WeightPredictor produces adaptive blending weights so that
    both experts contribute to both output modalities. Both buffers are updated
    autoregressively during training.
    """

    def __init__(self, cfg: LateFusionDenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sigma_data = cfg.sigma_data

        # RGB expert: conditions on RGB history, predicts noise in RGB space
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

        # Tactile expert: conditions on tactile history, predicts noise in tactile space
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

        # Shared conditioning modules for the weight predictor
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)
        hidden_per_step = cfg.cond_channels // cfg.num_steps_conditioning
        self.act_emb = nn.Sequential(
            nn.Linear(cfg.action_dim, hidden_per_step),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.weight_predictor = WeightPredictor(cfg.cond_channels)

        self.alpha = cfg.loss_alpha
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.model_rgb.noise_emb.weight.device

    # ------------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------------

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None, "setup_training called twice"

        def sample_sigma(n: int, device: torch.device) -> Tensor:
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    # ------------------------------------------------------------------
    # EDM preconditioning
    # ------------------------------------------------------------------

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
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
    # Per-expert calls
    # ------------------------------------------------------------------

    def _call_rgb_expert(
        self,
        noisy_next_rgb: Tensor,  # (B, 3, H, W)
        prev_rgb: Tensor,
        act: Tensor,
        cs: Conditioners,
    ) -> Tensor:
        return self.model_rgb(
            noisy_next_rgb * cs.c_in,
            cs.c_noise,
            cs.c_noise_cond,
            prev_rgb / self.sigma_data,
            act,
        )

    def _call_tac_expert(
        self,
        noisy_next_tac: Tensor,  # (B, 3, H, W) — noisy tactile target (not RGB)
        prev_tac: Tensor,
        act: Tensor,
        cs: Conditioners,
    ) -> Tensor:
        return self.model_tac(
            noisy_next_tac * cs.c_in,
            cs.c_noise,
            cs.c_noise_cond,
            prev_tac / self.sigma_data,
            act,
        )

    # ------------------------------------------------------------------
    # Output wrapping
    # ------------------------------------------------------------------

    @torch.no_grad()
    def wrap_model_output(self, noisy: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy + cs.c_out * model_output
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    # ------------------------------------------------------------------
    # Shared conditioning vector for weight predictor
    # ------------------------------------------------------------------

    def _compute_wp_cond(self, c_noise: Tensor, c_noise_cond: Tensor, act: Tensor) -> Tensor:
        """Compute the conditioning vector fed to the weight predictor.

        Args:
            c_noise:      (B,) — log-sigma / 4 (squeezed from Conditioners)
            c_noise_cond: (B,) — log-sigma_cond / 4
            act:          (B, n, action_dim)
        """
        act_emb = self.act_emb(act)  # (B, cond_channels)
        return self.cond_proj(
            self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb
        )  # (B, cond_channels)

    # ------------------------------------------------------------------
    # Inference denoising
    # ------------------------------------------------------------------

    @torch.no_grad()
    def denoise_composed(
        self,
        noisy_next_rgb: Tensor,
        noisy_next_tac: Tensor,
        sigma: Tensor,
        sigma_cond: Optional[Tensor],
        prev_rgb: Tensor,
        prev_tac: Tensor,
        act: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Single denoising step using both experts + weight predictor.

        Returns:
            denoised_rgb: (B, 3, H, W)
            denoised_tac: (B, 3, H, W)
        """
        cs = self.compute_conditioners(sigma, sigma_cond)

        # Weight predictor conditioning
        c_noise_1d = cs.c_noise.view(-1)
        c_noise_cond_1d = cs.c_noise_cond.view(-1)
        wp_cond = self._compute_wp_cond(c_noise_1d, c_noise_cond_1d, act)
        w_for_rgb, w_for_tac = self.weight_predictor(wp_cond)

        out_rgb = self._call_rgb_expert(noisy_next_rgb, prev_rgb, act, cs)
        out_tac = self._call_tac_expert(noisy_next_tac, prev_tac, act, cs)

        b = out_rgb.shape[0]
        w_rr = w_for_rgb[:, 0].view(b, 1, 1, 1)
        w_tr = w_for_rgb[:, 1].view(b, 1, 1, 1)
        w_rt = w_for_tac[:, 0].view(b, 1, 1, 1)
        w_tt = w_for_tac[:, 1].view(b, 1, 1, 1)

        composed_rgb = w_rr * out_rgb + w_tr * out_tac
        composed_tac = w_rt * out_rgb + w_tt * out_tac

        return (
            self.wrap_model_output(noisy_next_rgb, composed_rgb, cs),
            self.wrap_model_output(noisy_next_tac, composed_tac, cs),
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Autoregressive multi-step training forward pass over both modalities.

        batch keys:
          'front'   : (B, T, 3, H, W) — RGB frames in [-1, 1]
          'tactile' : (B, T, 3, H, W) — tactile frames in [-1, 1]
          'action'  : (B, T, 7)        — actions in [-1, 1]

        For each future step i:
          * model_rgb sees noisy RGB target + rolling prev_rgb
          * model_tac sees noisy tactile target + rolling prev_tac
          * WeightPredictor blends both experts for each output modality
          * Loss = alpha * MSE(composed_rgb, target_rgb) + (1-alpha) * MSE(composed_tac, target_tac)
          * Both RGB and tactile buffers are updated with their composed predictions
        """
        front = batch['front'].to(device)     # (B, T, 3, H, W)
        tactile = batch['tactile'].to(device)  # (B, T, 3, H, W)
        act = batch['action'].to(device)       # (B, T, 7)

        b, t, c, h, w = front.size()
        n = self.cfg.num_steps_conditioning
        seq_length = t - n

        all_rgb = front.clone()
        all_tac = tactile.clone()  # both roll out autoregressively

        total_loss = torch.tensor(0.0, device=device)
        total_loss_rgb = 0.0
        total_loss_tac = 0.0

        for i in range(seq_length):
            prev_rgb = all_rgb[:, i: n + i].reshape(b, n * c, h, w)
            prev_tac = all_tac[:, i: n + i].reshape(b, n * c, h, w)
            prev_act = act[:, i: n + i]
            target_rgb = all_rgb[:, n + i]
            target_tac = all_tac[:, n + i]

            # Noise both conditioning streams
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, device)
                prev_rgb_noised = self.apply_noise(prev_rgb, sigma_cond, self.cfg.sigma_offset_noise)
                prev_tac_noised = self.apply_noise(prev_tac, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None
                prev_rgb_noised = prev_rgb
                prev_tac_noised = prev_tac

            # Sample one noise level and corrupt both targets
            sigma = self.sample_sigma_training(b, device)
            noisy_rgb = self.apply_noise(target_rgb, sigma, self.cfg.sigma_offset_noise)
            noisy_tac = self.apply_noise(target_tac, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma, sigma_cond)

            # Weight predictor conditioning (uses 1-D c_noise values)
            c_noise_1d = cs.c_noise.view(b)
            c_noise_cond_1d = cs.c_noise_cond.view(b)
            wp_cond = self._compute_wp_cond(c_noise_1d, c_noise_cond_1d, prev_act)
            w_for_rgb, w_for_tac = self.weight_predictor(wp_cond)

            # Expert raw outputs
            out_rgb = self._call_rgb_expert(noisy_rgb, prev_rgb_noised, prev_act, cs)
            out_tac = self._call_tac_expert(noisy_tac, prev_tac_noised, prev_act, cs)

            # Adaptive blending: each weight pair sums to 1 (softmax)
            w_rr = w_for_rgb[:, 0].view(b, 1, 1, 1)  # RGB expert → RGB output
            w_tr = w_for_rgb[:, 1].view(b, 1, 1, 1)  # tac expert → RGB output
            w_rt = w_for_tac[:, 0].view(b, 1, 1, 1)  # RGB expert → tac output
            w_tt = w_for_tac[:, 1].view(b, 1, 1, 1)  # tac expert → tac output

            composed_rgb = w_rr * out_rgb + w_tr * out_tac
            composed_tac = w_rt * out_rgb + w_tt * out_tac

            # EDM targets
            target_rgb_edm = (target_rgb - cs.c_skip * noisy_rgb) / cs.c_out
            target_tac_edm = (target_tac - cs.c_skip * noisy_tac) / cs.c_out

            loss_rgb = F.mse_loss(composed_rgb, target_rgb_edm)
            loss_tac = F.mse_loss(composed_tac, target_tac_edm)
            step_loss = self.alpha * loss_rgb + (1.0 - self.alpha) * loss_tac
            total_loss = total_loss + step_loss
            total_loss_rgb += loss_rgb.item()
            total_loss_tac += loss_tac.item()

            # Update both rolling buffers with composed denoised outputs
            with torch.no_grad():
                denoised_rgb = self.wrap_model_output(noisy_rgb, composed_rgb, cs)
                denoised_tac = self.wrap_model_output(noisy_tac, composed_tac, cs)
            all_rgb[:, n + i] = denoised_rgb.detach()
            all_tac[:, n + i] = denoised_tac.detach()

        total_loss = total_loss / seq_length
        metrics = {
            "loss": total_loss.item(),
            "loss_rgb": total_loss_rgb / seq_length,
            "loss_tac": total_loss_tac / seq_length,
        }
        return total_loss, metrics
