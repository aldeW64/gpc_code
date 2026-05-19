"""Middle-Fusion multimodal LatentWorldModel for ManiFEEL RGB + tactile.

Fusion strategy
---------------
Two separate CNN encoders (one per modality) map each 3-channel input to an
intermediate latent.  A lightweight CrossModalAttention block then fuses the
two intermediate latents at the bottleneck.  The fused latent is passed to:
  - CMLatentDynamics for dynamics training (Stage 2)
  - CMDecoder for image reconstruction (Stage 1)

Only the RGB modality is decoded at inference for validation metrics; the
tactile stream influences dynamics through the fused latent.

Architecture:
  encoder_rgb:    Conv2d(3  -> latent_ch) + downsamples
  encoder_tac:    Conv2d(3  -> latent_ch) + downsamples  (independent)
  cross_attn:     CrossModalAttention(latent_ch) — zero-initialised, identity start
  fuse_proj:      Conv2d(2*latent_ch -> latent_ch)  — merges after attention
  dynamics:       CMLatentDynamics on fused latent
  decoder:        CMDecoder(x_shape=(3,H,W))  — decodes RGB only

Training commands (run from interactive_world_sim/ directory):

  # Stage 1: encoder + decoder (~200k steps)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_middle_fusion \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_middle_stage1

  # Stage 2: dynamics (replace <date>/<time>/<step> with actual values)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_middle_fusion \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_middle_stage2
"""

import os
import tracemalloc
from typing import Any, Callable

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from interactive_world_sim.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.common.metrics import (
    FrechetInceptionDistance,
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from interactive_world_sim.algorithms.models.cm_decoder import CMDecoder
from interactive_world_sim.algorithms.models.utils import EinopsWrapper
from interactive_world_sim.utils.cm_utils import DDPMScheduler
from interactive_world_sim.utils.logging_utils import (
    get_validation_metrics_for_videos,
    log_video,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


class CrossModalAttention(nn.Module):
    """Lightweight cross-modal attention between two latent feature maps.

    Zero-initialised output projection so that training starts from the
    identity (i.e., the RGB latent is unchanged at step 0).

    Args:
        dim: channel dimension of each latent feature map
        heads: number of attention heads
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = max(1, dim // heads)
        inner = self.heads * self.head_dim

        # RGB queries, tactile keys/values
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        # Zero-init: starts as identity residual
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self, rgb_feat: torch.Tensor, tac_feat: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend RGB (query) from tactile (key/value).

        Args:
            rgb_feat: (B, C, H, W)
            tac_feat: (B, C, H, W)

        Returns:
            rgb_feat updated by cross-attention: (B, C, H, W)
        """
        B, C, H, W = rgb_feat.shape
        # Flatten spatial dims for attention
        rgb_flat = rearrange(rgb_feat, "b c h w -> b (h w) c")
        tac_flat = rearrange(tac_feat, "b c h w -> b (h w) c")

        q = rearrange(self.q(rgb_flat), "b n (nh dh) -> b nh n dh", nh=self.heads)
        k = rearrange(self.k(tac_flat), "b n (nh dh) -> b nh n dh", nh=self.heads)
        v = rearrange(self.v(tac_flat), "b n (nh dh) -> b nh n dh", nh=self.heads)

        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, nh, N, N)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = attn_weights @ v  # (B, nh, N, dh)
        out = rearrange(out, "b nh n dh -> b n (nh dh)")
        out = self.out_proj(out)  # (B, N, C)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return rgb_feat + out


def _make_encoder(in_channels: int, latent_ch: int, num_downsample: int) -> nn.Sequential:
    """Build a stride-2 conv encoder."""
    layers = [nn.Conv2d(in_channels, latent_ch, 3, padding=1)]
    for _ in range(num_downsample):
        layers.extend(
            [
                nn.SiLU(),
                nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, stride=2),
            ]
        )
    return nn.Sequential(*layers)


class LatentWorldModelMiddleFusion(BasePytorchAlgo):
    """Middle-fusion multimodal LatentWorldModel.

    Separate CNN encoders per modality; latents fused via CrossModalAttention
    + channel-wise projection before passing to shared dynamics and decoder.
    Only RGB is decoded at inference (Stage 1 / val rendering).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = cfg.metrics
        self.num_latent_channel = cfg.num_latent_channel
        self.num_latent_downsample = cfg.num_latent_downsample
        self.training_stage = cfg.training_stage
        assert self.training_stage in [1, 2, 3], "Invalid training stage"
        self.load_ae = cfg.load_ae if "load_ae" in cfg else None
        super().__init__(cfg)
        self.normalizer = LinearNormalizer()
        self.validation_step_outputs: list = []
        self.validation_metrics: dict = {}
        self.timesteps: int = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.obs_keys = list(cfg.obs_keys)  # ["rgb", "tactile"]
        self.val_render = cfg.val_render
        self.clip_noise = self.cfg.diffusion.clip_noise
        self.guidance_scale = self.cfg.guidance_scale
        self.n_tokens = self.cfg.n_frames
        self.mask_prev_action = (
            cfg.mask_prev_action if "mask_prev_action" in cfg else False
        )
        # For encoder_forward normalisation: 1 view (fused latent treated as one block)
        self.num_views = 1

        self.latent_resolution = cfg.latent_resolution
        self.noise_scheduler: DDPMScheduler = hydra.utils.instantiate(
            cfg.noise_scheduler
        )

        self.debug = False
        self.lr_scheduler = cfg.lr_scheduler if "lr_scheduler" in cfg else "linear"
        self.sampling_strategy = (
            cfg.sampling_strategy if "sampling_strategy" in cfg else "uniform"
        )
        self.prev_frame_noise_scale = (
            cfg.prev_frame_noise_scale if "prev_frame_noise_scale" in cfg else 0.1
        )
        self.dyn_infer_steps = cfg.dyn_infer_steps if "dyn_infer_steps" in cfg else 1
        self.dec_infer_steps = cfg.dec_infer_steps if "dec_infer_steps" in cfg else 1
        self.last_frame_loss_only = (
            cfg.last_frame_loss_only if "last_frame_loss_only" in cfg else False
        )
        self.robust_latent = cfg.robust_latent if "robust_latent" in cfg else False

    def _build_model(self) -> None:
        latent_ch = self.num_latent_channel
        n_down = self.num_latent_downsample

        # Separate encoders for RGB and tactile (each 3-ch input)
        self.encoder_rgb = _make_encoder(3, latent_ch, n_down)
        self.encoder_tac = _make_encoder(3, latent_ch, n_down)

        # Cross-modal attention: tactile informs RGB latent
        attn_heads = getattr(self.cfg, "cross_attn_heads", 4)
        self.cross_attn = CrossModalAttention(dim=latent_ch, heads=attn_heads)

        # Project concatenated latents [rgb_attn | tac] -> latent_ch
        self.fuse_proj = nn.Conv2d(latent_ch * 2, latent_ch, kernel_size=1)

        # Decoder operates on 3-ch RGB only
        rgb_x_shape = list(self.cfg.x_shape)
        # Override to 3 channels (single modality decode)
        rgb_x_shape[0] = 3
        self.decoder: CMDecoder = CMDecoder(
            tuple(rgb_x_shape),
            self.cfg.latent_dim,
            self.cfg.diffusion,
            dtype=self.dtype,
        )

        # Dynamics on fused latent
        self.dynamics: EinopsWrapper = EinopsWrapper(
            from_shape="f b c h w",
            to_shape="b c f h w",
            module=hydra.utils.instantiate(self.cfg.dynamics),
        )

        if self.load_ae is not None:
            load_ae_dir = os.path.dirname(os.path.dirname(self.load_ae))
            cfg_path = f"{load_ae_dir}/.hydra/config.yaml"
            cfg_cp = OmegaConf.load(cfg_path)
            cfg_cp.algorithm.load_ae = None
            diffae = LatentWorldModelMiddleFusion.load_from_checkpoint(
                self.load_ae,
                cfg=cfg_cp.algorithm,
                map_location=self.device,
                weights_only=False,
            )
            self.encoder_rgb.load_state_dict(diffae.encoder_rgb.state_dict())
            self.encoder_tac.load_state_dict(diffae.encoder_tac.state_dict())
            self.cross_attn.load_state_dict(diffae.cross_attn.state_dict())
            self.fuse_proj.load_state_dict(diffae.fuse_proj.state_dict())
            if self.training_stage == 3:
                self.dynamics.load_state_dict(diffae.dynamics.state_dict())
            self.decoder.load_state_dict(diffae.decoder.state_dict())

        self.validation_fid_model = (
            FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        )
        self.validation_lpips_model = (
            LearnedPerceptualImagePatchSimilarity()
            if "lpips" in self.metrics
            else None
        )
        self.validation_fvd_model: FrechetVideoDistance = (
            FrechetVideoDistance() if "fvd" in self.metrics else None
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def configure_optimizers(self) -> dict:
        if self.training_stage == 1:
            enc_params = (
                list(self.encoder_rgb.parameters())
                + list(self.encoder_tac.parameters())
                + list(self.cross_attn.parameters())
                + list(self.fuse_proj.parameters())
            )
            param_groups = [
                {"params": enc_params, "lr": self.cfg.lr},
                {"params": self.decoder.parameters(), "lr": self.cfg.lr},
            ]
        elif self.training_stage == 2:
            param_groups = [
                {"params": self.dynamics.parameters(), "lr": self.cfg.lr},
            ]
        elif self.training_stage == 3:
            param_groups = [
                {"params": self.decoder.parameters(), "lr": self.cfg.lr * 0.1},
            ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        if self.lr_scheduler == "linear":
            lr_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=self.cfg.warmup_steps,
            )
        elif self.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=50000,
                verbose=True,
                threshold=1e-3,
                threshold_mode="rel",
            )
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler} not included")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "training/loss",
                "strict": True,
                "name": "lr_scheduler",
            },
        }

    def _encode_fused(
        self, rgb_flat: torch.Tensor, tac_flat: torch.Tensor
    ) -> torch.Tensor:
        """Encode and fuse both modalities into a single latent.

        Args:
            rgb_flat: (B*T, 3, H, W)
            tac_flat: (B*T, 3, H, W)

        Returns:
            z_fused: (B*T, latent_ch, H_l, W_l)
        """
        z_rgb = self.encoder_rgb(rgb_flat)   # (B*T, latent_ch, H_l, W_l)
        z_tac = self.encoder_tac(tac_flat)   # (B*T, latent_ch, H_l, W_l)

        # Cross-modal: RGB queries, tactile keys/values
        z_rgb_attended = self.cross_attn(z_rgb, z_tac)

        # Fuse: concatenate + project
        z_fused = self.fuse_proj(torch.cat([z_rgb_attended, z_tac], dim=1))

        # L2-normalise
        z_fused = z_fused / (torch.norm(z_fused, dim=1, keepdim=True) + 1e-8)
        return z_fused

    def encoder_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compatibility shim: expects concatenated 6-ch input (rgb | tac).

        Splits on the channel dimension and calls _encode_fused.
        """
        rgb = obs[:, :3]
        tac = obs[:, 3:6]
        return self._encode_fused(rgb, tac)

    def optimizer_step(
        self,
        epoch: dict,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Callable,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if self.training_stage == 2:
            for name, param in self.dynamics.named_parameters():
                if (
                    param.requires_grad
                    and (param.grad is not None)
                    and torch.isnan(param.grad).any()
                ):
                    print(f"NaN in gradient of {name}")
                    exit()

    def _forward(
        self,
        model: Any,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        stop_time: torch.Tensor,
        external_cond: Any = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        assert (timestep >= stop_time).all()
        assert (timestep[-1] > stop_time[-1]).all()
        denoise = lambda x, t, s: model(x, t, s, external_cond=external_cond)
        return self.noise_scheduler.CTM_calc_out(
            denoise, sample, timestep, stop_time, clamp=clamp
        )

    @torch.no_grad()
    def dynamics_forward(
        self, z_0: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Dynamics forward (operates on fused latent)."""
        z_0 = rearrange(z_0, "b t c h w -> t b c h w")
        action = rearrange(action, "b t c -> t b c")
        T_hist = z_0.shape[0]
        T_act = action.shape[0] - T_hist
        chunk_size = 1
        curr_end = T_hist + chunk_size
        total_frames = T_hist + T_act
        xs_pred = z_0.clone()
        batch_size = z_0.shape[1]

        while curr_end <= total_frames:
            horizon = chunk_size
            chunk = torch.randn(
                (horizon, batch_size, *z_0.shape[2:]),
                device=self.device,
                dtype=self.dtype,
            )
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)
            curr_start = max(0, curr_end - self.n_tokens)
            clean_t = (
                torch.ones(
                    (xs_pred[curr_start:].shape[0] - 1,), device=self.device
                )
                * self.noise_scheduler.stabilization_level
            )
            timesteps = torch.linspace(
                self.noise_scheduler.timesteps - 1,
                0,
                self.dyn_infer_steps + 1,
                device=z_0.device,
            )
            action_chunk = action[curr_start:curr_end]
            if self.mask_prev_action:
                action_chunk[:-1] = 0

            for step_i in range(self.dyn_infer_steps):
                t = timesteps[step_i].unsqueeze(0)
                s = timesteps[step_i + 1].unsqueeze(0)
                t = torch.cat([clean_t, t], 0)
                t = torch.tile(t[:, None], (1, xs_pred.shape[1]))
                s = torch.cat([clean_t, s], 0)
                s = torch.tile(s[:, None], (1, xs_pred.shape[1]))
                t = t.long()
                s = s.long()
                xs_pred_updated = self._forward(
                    self.dynamics,
                    xs_pred[curr_start:],
                    t,
                    s,
                    external_cond=action_chunk,
                )
                if self.last_frame_loss_only:
                    xs_pred[-1:] = xs_pred_updated[-1:]
                else:
                    xs_pred[curr_start:] = xs_pred_updated

            curr_end += horizon

        xs_pred = xs_pred / (torch.norm(xs_pred, dim=2, keepdim=True) + 1e-8)
        xs_pred = rearrange(xs_pred[T_hist:], "t b c h w -> b t c h w")
        return xs_pred

    def _generate_noise_levels(
        self, xs: torch.Tensor, cm_steps: int = -1
    ) -> torch.Tensor:
        num_frames, batch_size, *_ = xs.shape
        if self.sampling_strategy == "uniform":
            last_t = torch.randint(2, self.timesteps, (batch_size,))
            last_s = torch.cat(
                [torch.randint(1, int(t_i.item()), (1,)) for t_i in last_t]
            )
            last_t = last_t.unsqueeze(0).to(xs.device)
            last_s = last_s.unsqueeze(0).to(xs.device)
        elif self.sampling_strategy == "terminal_only":
            last_t = torch.ones((batch_size,)) * (self.timesteps - 1)
            last_t = last_t.unsqueeze(0).to(xs.device)
            if cm_steps == 1:
                last_s = torch.zeros((batch_size,))
                last_s = last_s.unsqueeze(0).to(xs.device)
            else:
                intermediate_s = np.linspace(
                    0, self.timesteps - 1, cm_steps + 1, dtype=int
                )
                s_val = np.random.choice(intermediate_s[1:-1], size=(batch_size,))
                last_s = torch.ones((batch_size,)) * s_val
                last_s = last_s.unsqueeze(0).to(xs.device)

        prev_noise_levels = torch.randint(
            1,
            int(self.timesteps * self.prev_frame_noise_scale),
            (num_frames - 1, batch_size),
            device=xs.device,
        )
        t = torch.cat([prev_noise_levels, last_t], 0)
        s = torch.cat([prev_noise_levels, last_s], 0)
        return t.long(), s.long()

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None
        if batch_idx % 1000 == 0:
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.compare_to(
                self.tracemalloc_snapshot, "lineno"
            )
            print(f"\n[ Top 10 memory diff at step {batch_idx} ]")
            for stat in top_stats[:10]:
                print(stat)
        assert "valid_mask" not in batch

        rgb_obs = self.normalizer["rgb"].normalize(batch["obs"]["rgb"])   # (B,T,3,H,W)
        tac_obs = self.normalizer["tactile"].normalize(batch["obs"]["tactile"])
        action = self.normalizer["action"].normalize(batch["action"])

        rgb_obs = rgb_obs.float()
        tac_obs = tac_obs.float()
        action = action.float()

        B, T = rgb_obs.shape[:2]
        rgb_flat = rearrange(rgb_obs, "b t c h w -> (b t) c h w")
        tac_flat = rearrange(tac_obs, "b t c h w -> (b t) c h w")
        # RGB is target for reconstruction
        xs = rgb_flat  # (B*T, 3, H, W)

        output_dict = {}

        if self.training_stage == 1:
            z = self._encode_fused(rgb_flat, tac_flat)
            if self.robust_latent:
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)
            weights_t = self.noise_scheduler.get_weights(t)[0]
            weights_s = self.noise_scheduler.get_weights(s)[0]
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )
            noisy_xs_t = noisy_xs_t.squeeze(0)
            noisy_xs_s = noisy_xs_s.squeeze(0)
            t = t.squeeze(0)
            s = s.squeeze(0)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(self.decoder, noisy_xs_t, t, s, external_cond=z)
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder, noisy_xs_s, s, u, external_cond=z
                )

            loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
            weights_t = weights_t.view(
                *weights_t.shape, *((1,) * (loss_s.ndim - 1))
            )
            loss_s = loss_s * weights_t
            if self.dec_infer_steps > 1:
                loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                weights_s = weights_s.view(
                    *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_u = loss_u * weights_s
                loss = (loss_s + loss_u).mean()
            else:
                loss = loss_s.mean()

            self.log("training/rec_loss", loss)
            return {"loss": loss}

        elif self.training_stage == 2:
            with torch.no_grad():
                z = self._encode_fused(rgb_flat, tac_flat)
            z = rearrange(z, "(b t) c h w -> t b c h w", b=B)
            action = rearrange(action, "b t a -> t b a")

            t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
            weights_t = self.noise_scheduler.get_weights(t)
            weights_s = self.noise_scheduler.get_weights(s)
            noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)

            u = torch.zeros_like(t).to(self.device)
            if self.mask_prev_action:
                action[:-1] = 0
            pred_s = self._forward(
                self.dynamics, noisy_z_t, t, s, external_cond=action
            )
            if self.dyn_infer_steps > 1:
                pred_u = self._forward(
                    self.dynamics, noisy_z_s, s, u, external_cond=action
                )

            loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
            weights_t = weights_t.view(
                *weights_t.shape, *((1,) * (loss_s.ndim - 2))
            )
            loss_s = loss_s * weights_t
            if self.dyn_infer_steps > 1:
                loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
                weights_s = weights_s.view(
                    *weights_s.shape, *((1,) * (loss_s.ndim - 2))
                )
                loss_u = loss_u * weights_s
                loss = (loss_s + loss_u).mean()
            else:
                loss = loss_s.mean()

            output_dict["loss"] = loss
            self.log("training/loss", loss)
            for key in output_dict:
                self.log(f"training/{key}", output_dict[key])

        elif self.training_stage == 3:
            with torch.no_grad():
                z = self._encode_fused(rgb_flat, tac_flat)
                z += torch.randn_like(z) * 0.02

            t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)
            weights_t = self.noise_scheduler.get_weights(t)[0]
            weights_s = self.noise_scheduler.get_weights(s)[0]
            noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(
                xs[None], t, s
            )
            noisy_xs_t = noisy_xs_t.squeeze(0)
            noisy_xs_s = noisy_xs_s.squeeze(0)
            t = t.squeeze(0)
            s = s.squeeze(0)

            u = torch.zeros_like(t).to(self.device)
            pred_s = self._forward(self.decoder, noisy_xs_t, t, s, external_cond=z)
            if self.dec_infer_steps > 1:
                pred_u = self._forward(
                    self.decoder, noisy_xs_s, s, u, external_cond=z
                )

            loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
            weights_t = weights_t.view(
                *weights_t.shape, *((1,) * (loss_s.ndim - 1))
            )
            loss_s = loss_s * weights_t
            if self.dec_infer_steps > 1:
                loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
                weights_s = weights_s.view(
                    *weights_s.shape, *((1,) * (loss_s.ndim - 1))
                )
                loss_u = loss_u * weights_s
                loss = (loss_s + loss_u).mean()
            else:
                loss = loss_s.mean()

            self.log("training/rec_loss", loss)
            return {"loss": loss}

        return output_dict

    def validation_step(
        self, batch: dict, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        rgb_obs = self.normalizer["rgb"].normalize(batch["obs"]["rgb"]).float()
        tac_obs = self.normalizer["tactile"].normalize(batch["obs"]["tactile"]).float()
        action = self.normalizer["action"].normalize(batch["action"]).float()

        B, T = rgb_obs.shape[:2]
        rgb_flat = rearrange(rgb_obs, "b t c h w -> (b t) c h w")
        tac_flat = rearrange(tac_obs, "b t c h w -> (b t) c h w")

        z_gt = self._encode_fused(rgb_flat, tac_flat)
        z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=B)

        if self.training_stage in [1, 3]:
            z_seq = z_gt
        elif self.training_stage == 2:
            z_0 = z_gt[:, 0]
            z_seq_ls = []
            z_last = z_0.clone()
            horizon = z_gt.shape[1]
            for i in range(1, action.shape[1], horizon):
                action_chunk = action[:, i : i + horizon]
                init_action_size = action_chunk.shape[1]
                if init_action_size < horizon:
                    action_chunk = F.pad(
                        action_chunk,
                        (0, 0, 0, horizon - action_chunk.shape[1]),
                        mode="replicate",
                    )
                z_seq = self.dynamics_forward(z_last[:, None], action_chunk)
                z_seq = z_seq[:, :init_action_size]
                z_seq_ls.append(z_seq)
                z_last = z_seq[:, -1].clone()
            z_seq = torch.cat(z_seq_ls, 1)
            z_seq = torch.cat([z_0.unsqueeze(1), z_seq], 1)
            val_loss = F.mse_loss(z_seq, z_gt, reduction="none")[:, 1:].mean()
            self.log(f"{namespace}/dyn_loss", val_loss)
        else:
            z_seq = z_gt

        z_seq_flat = rearrange(z_seq, "b t c h w -> (b t) c h w")

        if self.val_render:
            # Render RGB only using the shared decoder
            xs_flat = rgb_flat  # (B*T, 3, H, W)
            resolution = xs_flat.shape[-1]
            xs_pred = torch.randn_like(xs_flat).to(device=self.device, dtype=self.dtype)
            schedules = np.linspace(
                self.timesteps - 1, 0, self.dec_infer_steps + 1
            )
            batch_sz = 50
            for j in range(0, xs_pred.shape[0], batch_sz):
                actual_bs = xs_pred[j : j + batch_sz].shape[0]
                for step_i in range(self.dec_infer_steps):
                    t = torch.tensor(
                        [schedules[step_i]], device=self.device
                    ).repeat(actual_bs).long()
                    s = torch.tensor(
                        [schedules[step_i + 1]], device=self.device
                    ).repeat(actual_bs).long()
                    xs_pred[j : j + batch_sz] = self._forward(
                        self.decoder,
                        xs_pred[j : j + batch_sz],
                        t,
                        s,
                        external_cond=z_seq_flat[j : j + batch_sz],
                    )
            # Unnormalise
            xs_pred = self.normalizer["rgb"].unnormalize(xs_pred).clamp(0, 1)
            xs_pred = rearrange(xs_pred, "(b t) c h w -> t b c h w", b=B)
            xs_gt = rearrange(batch["obs"]["rgb"], "b t c h w -> t b c h w")
            self.validation_step_outputs.append(
                (xs_pred.detach().cpu(), xs_gt.detach().cpu())
            )
        return

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def on_validation_epoch_end(self, namespace: str = "validation") -> None:
        if not self.validation_step_outputs:
            return
        xs_pred_ls, xs_ls = [], []
        for pred, gt in self.validation_step_outputs:
            xs_pred_ls.append(pred)
            xs_ls.append(gt)
        xs_pred = torch.cat(xs_pred_ls, 1)
        xs = torch.cat(xs_ls, 1)

        if self.logger:
            log_video(
                xs_pred,
                xs.clone(),
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred,
            xs,
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=self.validation_fvd_model,
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_outputs.clear()

    def on_train_start(self) -> None:
        tracemalloc.start()
        self.tracemalloc_snapshot = tracemalloc.take_snapshot()
