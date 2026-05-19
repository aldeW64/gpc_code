"""Late-Fusion multimodal LatentWorldModel for ManiFEEL RGB + tactile.

Fusion strategy
---------------
Two completely independent encoder–decoder–dynamics pipelines (one for RGB,
one for tactile) share NO weights.  At inference, the **dynamics predictions
are combined by a weighted average** in latent space before decoding.

Stage 1:
  - encoder_rgb + decoder_rgb: reconstruct RGB from RGB latent
  - encoder_tac + decoder_tac: reconstruct tactile from tactile latent
  - Training loss: alpha * loss_rgb + (1-alpha) * loss_tac   (default alpha=0.5)

Stage 2:
  - dynamics_rgb: predict future RGB latents conditioned on actions
  - dynamics_tac: predict future tactile latents conditioned on actions
  - At inference: z_pred = w_rgb * z_rgb + w_tac * z_tac  (composition weights)
  - Only RGB is decoded for metrics/visualisation

Architecture:
  encoder_rgb / encoder_tac:   Conv2d(3 -> latent_ch) + downsamples
  decoder_rgb / decoder_tac:   CMDecoder(x_shape=(3,H,W))
  dynamics_rgb / dynamics_tac: CMLatentDynamics
  composition_weights:         [w_rgb, w_tac]  (from cfg, default [0.7, 0.3])
  loss_alpha:                  weight on RGB loss in Stage 1  (default 0.5)

Training commands (run from interactive_world_sim/ directory):

  # Stage 1: dual encoder + decoder (~200k steps)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_late_fusion \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_late_stage1

  # Stage 2: dual dynamics (replace <date>/<time>/<step> with actual values)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_late_fusion \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_late_stage2
"""

import os
import tracemalloc
from typing import Any, Callable, List

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


def _make_encoder(in_channels: int, latent_ch: int, num_downsample: int) -> nn.Sequential:
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


def _make_dynamics(cfg: DictConfig) -> EinopsWrapper:
    return EinopsWrapper(
        from_shape="f b c h w",
        to_shape="b c f h w",
        module=hydra.utils.instantiate(cfg.dynamics),
    )


class LatentWorldModelLateFusion(BasePytorchAlgo):
    """Late-fusion multimodal LatentWorldModel.

    Two completely independent pipelines (RGB and tactile); combined only at
    the dynamics prediction stage via a weighted latent average.
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
        self.obs_keys: List[str] = list(cfg.obs_keys)  # ["rgb", "tactile"]
        self.val_render = cfg.val_render
        self.clip_noise = self.cfg.diffusion.clip_noise
        self.guidance_scale = self.cfg.guidance_scale
        self.n_tokens = self.cfg.n_frames
        self.mask_prev_action = (
            cfg.mask_prev_action if "mask_prev_action" in cfg else False
        )
        # Reported as 1 view (RGB) for metrics
        self.num_views = 1

        self.latent_resolution = cfg.latent_resolution
        self.noise_scheduler: DDPMScheduler = hydra.utils.instantiate(
            cfg.noise_scheduler
        )

        # Late-fusion-specific hyperparameters
        self.loss_alpha: float = float(cfg.get("loss_alpha", 0.5))
        composition_weights = cfg.get("composition_weights", [0.7, 0.3])
        self.w_rgb: float = float(composition_weights[0])
        self.w_tac: float = float(composition_weights[1])

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

        # Independent encoders
        self.encoder_rgb = _make_encoder(3, latent_ch, n_down)
        self.encoder_tac = _make_encoder(3, latent_ch, n_down)

        # Independent dynamics
        self.dynamics_rgb: EinopsWrapper = _make_dynamics(self.cfg)
        self.dynamics_tac: EinopsWrapper = _make_dynamics(self.cfg)

        # Independent decoders — both decode to 3-ch images
        rgb_x_shape = list(self.cfg.x_shape)
        rgb_x_shape[0] = 3
        x_shape_3ch = tuple(rgb_x_shape)

        self.decoder_rgb: CMDecoder = CMDecoder(
            x_shape_3ch,
            self.cfg.latent_dim,
            self.cfg.diffusion,
            dtype=self.dtype,
        )
        self.decoder_tac: CMDecoder = CMDecoder(
            x_shape_3ch,
            self.cfg.latent_dim,
            self.cfg.diffusion,
            dtype=self.dtype,
        )

        if self.load_ae is not None:
            load_ae_dir = os.path.dirname(os.path.dirname(self.load_ae))
            cfg_path = f"{load_ae_dir}/.hydra/config.yaml"
            cfg_cp = OmegaConf.load(cfg_path)
            cfg_cp.algorithm.load_ae = None
            diffae = LatentWorldModelLateFusion.load_from_checkpoint(
                self.load_ae,
                cfg=cfg_cp.algorithm,
                map_location=self.device,
                weights_only=False,
            )
            self.encoder_rgb.load_state_dict(diffae.encoder_rgb.state_dict())
            self.encoder_tac.load_state_dict(diffae.encoder_tac.state_dict())
            self.decoder_rgb.load_state_dict(diffae.decoder_rgb.state_dict())
            self.decoder_tac.load_state_dict(diffae.decoder_tac.state_dict())
            if self.training_stage == 3:
                self.dynamics_rgb.load_state_dict(diffae.dynamics_rgb.state_dict())
                self.dynamics_tac.load_state_dict(diffae.dynamics_tac.state_dict())

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
                + list(self.decoder_rgb.parameters())
                + list(self.decoder_tac.parameters())
            )
            param_groups = [{"params": enc_params, "lr": self.cfg.lr}]
        elif self.training_stage == 2:
            dyn_params = (
                list(self.dynamics_rgb.parameters())
                + list(self.dynamics_tac.parameters())
            )
            param_groups = [{"params": dyn_params, "lr": self.cfg.lr}]
        elif self.training_stage == 3:
            param_groups = [
                {
                    "params": (
                        list(self.decoder_rgb.parameters())
                        + list(self.decoder_tac.parameters())
                    ),
                    "lr": self.cfg.lr * 0.1,
                }
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

    def _encode(self, encoder: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Encode and L2-normalise."""
        z = encoder(x)
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-8)
        return z

    def optimizer_step(
        self,
        epoch: dict,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Callable,
    ) -> None:
        optimizer.step(closure=optimizer_closure)

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

    def _dynamics_forward_single(
        self,
        dynamics: EinopsWrapper,
        z_0: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Run one dynamics pipeline autoregressively."""
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
                    dynamics,
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

    @torch.no_grad()
    def dynamics_forward(
        self, z_0: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Combined dynamics: weighted average of RGB and tactile predictions.

        z_0 is assumed to be the RGB latent for this interface.
        """
        # For late fusion at inference: we pass in z_rgb_0 and compose
        # In this simplified interface, z_0 is z_rgb and we skip tactile
        # (no tactile history at inference time unless explicitly provided).
        return self._dynamics_forward_single(self.dynamics_rgb, z_0, action)

    @torch.no_grad()
    def dynamics_forward_bimodal(
        self,
        z_rgb_0: torch.Tensor,
        z_tac_0: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Combined dynamics with both latent streams."""
        z_rgb_pred = self._dynamics_forward_single(self.dynamics_rgb, z_rgb_0, action)
        z_tac_pred = self._dynamics_forward_single(self.dynamics_tac, z_tac_0, action)
        return self.w_rgb * z_rgb_pred + self.w_tac * z_tac_pred

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

    def _stage1_loss_for_modality(
        self,
        encoder: nn.Sequential,
        decoder: CMDecoder,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CM reconstruction loss for a single modality."""
        z = self._encode(encoder, xs)
        if self.robust_latent:
            z += torch.randn_like(z) * 0.02

        t, s = self._generate_noise_levels(xs[None], self.dec_infer_steps)
        weights_t = self.noise_scheduler.get_weights(t)[0]
        weights_s = self.noise_scheduler.get_weights(s)[0]
        noisy_xs_t, noisy_xs_s = self.noise_scheduler.add_noise_to_t_s(xs[None], t, s)
        noisy_xs_t = noisy_xs_t.squeeze(0)
        noisy_xs_s = noisy_xs_s.squeeze(0)
        t = t.squeeze(0)
        s = s.squeeze(0)
        u = torch.zeros_like(t).to(self.device)

        pred_s = self._forward(decoder, noisy_xs_t, t, s, external_cond=z)
        if self.dec_infer_steps > 1:
            pred_u = self._forward(decoder, noisy_xs_s, s, u, external_cond=z)

        loss_s = F.mse_loss(pred_s, noisy_xs_s.detach(), reduction="none")
        weights_t = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 1)))
        loss_s = loss_s * weights_t
        if self.dec_infer_steps > 1:
            loss_u = F.mse_loss(pred_u, xs.detach(), reduction="none")
            weights_s = weights_s.view(*weights_s.shape, *((1,) * (loss_s.ndim - 1)))
            loss_u = loss_u * weights_s
            return (loss_s + loss_u).mean()
        return loss_s.mean()

    def _stage2_loss_for_modality(
        self,
        dynamics: EinopsWrapper,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamics CM loss for a single modality."""
        t, s = self._generate_noise_levels(z, self.dyn_infer_steps)
        weights_t = self.noise_scheduler.get_weights(t)
        weights_s = self.noise_scheduler.get_weights(s)
        noisy_z_t, noisy_z_s = self.noise_scheduler.add_noise_to_t_s(z, t, s)
        u = torch.zeros_like(t).to(self.device)

        action_in = action.clone()
        if self.mask_prev_action:
            action_in[:-1] = 0

        pred_s = self._forward(dynamics, noisy_z_t, t, s, external_cond=action_in)
        if self.dyn_infer_steps > 1:
            pred_u = self._forward(dynamics, noisy_z_s, s, u, external_cond=action_in)

        loss_s = F.mse_loss(pred_s, noisy_z_s.detach(), reduction="none")
        weights_t = weights_t.view(*weights_t.shape, *((1,) * (loss_s.ndim - 2)))
        loss_s = loss_s * weights_t
        if self.dyn_infer_steps > 1:
            loss_u = F.mse_loss(pred_u, z.detach(), reduction="none")
            weights_s = weights_s.view(*weights_s.shape, *((1,) * (loss_s.ndim - 2)))
            loss_u = loss_u * weights_s
            return (loss_s + loss_u).mean()
        return loss_s.mean()

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

        rgb_obs = self.normalizer["rgb"].normalize(batch["obs"]["rgb"]).float()
        tac_obs = self.normalizer["tactile"].normalize(batch["obs"]["tactile"]).float()
        action = self.normalizer["action"].normalize(batch["action"]).float()

        B, T = rgb_obs.shape[:2]
        rgb_flat = rearrange(rgb_obs, "b t c h w -> (b t) c h w")
        tac_flat = rearrange(tac_obs, "b t c h w -> (b t) c h w")

        output_dict = {}

        if self.training_stage == 1:
            loss_rgb = self._stage1_loss_for_modality(
                self.encoder_rgb, self.decoder_rgb, rgb_flat
            )
            loss_tac = self._stage1_loss_for_modality(
                self.encoder_tac, self.decoder_tac, tac_flat
            )
            loss = self.loss_alpha * loss_rgb + (1 - self.loss_alpha) * loss_tac
            self.log("training/rec_loss_rgb", loss_rgb)
            self.log("training/rec_loss_tac", loss_tac)
            self.log("training/rec_loss", loss)
            return {"loss": loss}

        elif self.training_stage == 2:
            with torch.no_grad():
                z_rgb = self._encode(self.encoder_rgb, rgb_flat)
                z_tac = self._encode(self.encoder_tac, tac_flat)
            z_rgb = rearrange(z_rgb, "(b t) c h w -> t b c h w", b=B)
            z_tac = rearrange(z_tac, "(b t) c h w -> t b c h w", b=B)
            action_t = rearrange(action, "b t a -> t b a")

            loss_rgb = self._stage2_loss_for_modality(self.dynamics_rgb, z_rgb, action_t)
            loss_tac = self._stage2_loss_for_modality(self.dynamics_tac, z_tac, action_t)
            loss = self.loss_alpha * loss_rgb + (1 - self.loss_alpha) * loss_tac

            output_dict["loss"] = loss
            self.log("training/loss", loss)
            self.log("training/dyn_loss_rgb", loss_rgb)
            self.log("training/dyn_loss_tac", loss_tac)
            for key in output_dict:
                self.log(f"training/{key}", output_dict[key])

        elif self.training_stage == 3:
            with torch.no_grad():
                z_rgb = self._encode(self.encoder_rgb, rgb_flat)
                z_rgb += torch.randn_like(z_rgb) * 0.02
                z_tac = self._encode(self.encoder_tac, tac_flat)
                z_tac += torch.randn_like(z_tac) * 0.02

            # Fine-tune decoders with perturbed latents
            loss_rgb = self._stage1_loss_for_modality(
                self.encoder_rgb, self.decoder_rgb, rgb_flat
            )
            loss_tac = self._stage1_loss_for_modality(
                self.encoder_tac, self.decoder_tac, tac_flat
            )
            loss = self.loss_alpha * loss_rgb + (1 - self.loss_alpha) * loss_tac
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

        z_rgb_gt = self._encode(self.encoder_rgb, rgb_flat)
        z_rgb_gt = rearrange(z_rgb_gt, "(b t) c h w -> b t c h w", b=B)

        if self.training_stage in [1, 3]:
            z_seq = z_rgb_gt
        elif self.training_stage == 2:
            z_tac_gt = self._encode(self.encoder_tac, tac_flat)
            z_tac_gt = rearrange(z_tac_gt, "(b t) c h w -> b t c h w", b=B)

            z_rgb_0 = z_rgb_gt[:, 0:1]
            z_tac_0 = z_tac_gt[:, 0:1]

            z_rgb_seq_ls = []
            z_tac_seq_ls = []
            z_rgb_last = z_rgb_0.clone()
            z_tac_last = z_tac_0.clone()
            horizon = z_rgb_gt.shape[1]

            for i in range(1, action.shape[1], horizon):
                action_chunk = action[:, i : i + horizon]
                init_action_size = action_chunk.shape[1]
                if init_action_size < horizon:
                    action_chunk = F.pad(
                        action_chunk,
                        (0, 0, 0, horizon - action_chunk.shape[1]),
                        mode="replicate",
                    )
                z_rgb_pred = self._dynamics_forward_single(
                    self.dynamics_rgb, z_rgb_last, action_chunk
                )
                z_tac_pred = self._dynamics_forward_single(
                    self.dynamics_tac, z_tac_last, action_chunk
                )
                z_rgb_seq_ls.append(z_rgb_pred[:, :init_action_size])
                z_tac_seq_ls.append(z_tac_pred[:, :init_action_size])
                z_rgb_last = z_rgb_pred[:, -1:].clone()
                z_tac_last = z_tac_pred[:, -1:].clone()

            z_rgb_seq = torch.cat(z_rgb_seq_ls, 1)
            z_rgb_seq = torch.cat([z_rgb_gt[:, 0:1], z_rgb_seq], 1)

            # Composed prediction for metrics
            z_tac_seq = torch.cat(z_tac_seq_ls, 1)
            z_tac_seq = torch.cat([z_tac_gt[:, 0:1], z_tac_seq], 1)
            z_composed = self.w_rgb * z_rgb_seq + self.w_tac * z_tac_seq
            z_seq = z_composed

            val_loss_rgb = F.mse_loss(
                z_rgb_seq, z_rgb_gt, reduction="none"
            )[:, 1:].mean()
            self.log(f"{namespace}/dyn_loss_rgb", val_loss_rgb)
        else:
            z_seq = z_rgb_gt

        z_seq_flat = rearrange(z_seq, "b t c h w -> (b t) c h w")

        if self.val_render:
            resolution = rgb_flat.shape[-1]
            xs_pred = torch.randn(
                rgb_flat.shape[0], 3, resolution, resolution,
                device=self.device, dtype=self.dtype
            )
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
                        self.decoder_rgb,
                        xs_pred[j : j + batch_sz],
                        t,
                        s,
                        external_cond=z_seq_flat[j : j + batch_sz],
                    )
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
