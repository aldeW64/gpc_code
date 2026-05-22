from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# denoiser.py uses `from data import Batch` and inner_model.py uses
# `from diffusion.blocks import ...` — both are absolute imports that only
# resolve when gpc_rank_evaluation/ itself is on sys.path.
_GPC_RANK_DIR = str(Path(__file__).absolute().parents[1] / "gpc_rank_evaluation")
if _GPC_RANK_DIR not in sys.path:
    sys.path.insert(0, _GPC_RANK_DIR)

from gpc_rank_evaluation.diffusion.denoiser import Denoiser
from gpc_rank_evaluation.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

_ACTION_MIN = np.array([0.0, 0.0], dtype=np.float32)
_ACTION_MAX = np.array([511.0, 511.0], dtype=np.float32)

_NUM_COND = 4  # num_steps_conditioning — matches eval_baseline.py


# Architecture config mirrors eval_baseline.py exactly.
@dataclass
class _SigmaCfg:
    loc = -1.2
    scale = 1.2
    sigma_min = 2e-3
    sigma_max = 20.0


@dataclass
class _InnerModelCfg:
    img_channels = 3
    num_steps_conditioning = _NUM_COND
    cond_channels = 256
    depths = [2, 2, 2, 2]
    channels = [96, 96, 96, 96]
    attn_depths = [0, 0, 1, 1]
    num_actions = _NUM_COND
    is_upsampler = None


@dataclass
class _DenoiserCfg:
    inner_model = _InnerModelCfg()
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True
    upsampling_factor = None


@dataclass
class GpcWorldModelConfig:
    ckpt_path: str
    num_diffusion_steps: int = 3


class GpcWorldModelAdapter:
    """
    Wraps the GPC diffusion world model (denoiser.pth from world_model_checkpoint).

    Performs autoregressive single-step prediction to roll out a trajectory.
    Input history images must be float32 in [0,1]; output is also [0,1].
    """

    num_history: int = _NUM_COND  # number of context frames required

    def __init__(self, cfg: GpcWorldModelConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        denoiser = Denoiser(_DenoiserCfg())
        denoiser.setup_training(_SigmaCfg())
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        denoiser.load_state_dict(state)
        denoiser.to(self.device).eval()

        self.sampler = DiffusionSampler(
            denoiser, DiffusionSamplerConfig(num_steps_denoising=cfg.num_diffusion_steps)
        )
        print(f"[gpc_wm] loaded GPC world model from {cfg.ckpt_path}", flush=True)

    @torch.no_grad()
    def _rollout(
        self,
        history_images_chw: np.ndarray,  # (N >= 4, 3, H, W) float32 in [0,1]
        actions: np.ndarray,              # (T, 2)  float32 in env pixel coords [0,511]
    ) -> list:
        """Core autoregressive rollout. Returns list of T predicted frames (3,H,W) float32 [0,1]."""
        T = len(actions)
        n = _NUM_COND
        assert history_images_chw.shape[0] >= n, (
            f"GpcWorldModelAdapter needs >= {n} history frames, got {history_images_chw.shape[0]}"
        )

        norm_acts = (actions.astype(np.float32) - _ACTION_MIN) / (_ACTION_MAX - _ACTION_MIN)
        act_buf = np.zeros((n + T, 2), dtype=np.float32)
        act_buf[n:] = norm_acts

        window: list[np.ndarray] = list(history_images_chw[-n:])
        predicted: list[np.ndarray] = []

        for t in range(T):
            prev_imgs = np.stack(window[-n:])  # (n, 3, H, W)
            img_t = torch.from_numpy(prev_imgs).unsqueeze(0).to(self.device, dtype=torch.float32)
            act_t = torch.from_numpy(act_buf[t: t + n]).unsqueeze(0).to(self.device, dtype=torch.float32)

            pred, _ = self.sampler.sample(img_t, act_t)  # (1, 3, H, W) in [-1,1]

            pred_01 = ((pred.squeeze(0).cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
            window.append(pred_01)
            predicted.append(pred_01)

        return predicted

    @torch.no_grad()
    def rollout_final_image(
        self,
        history_images_chw: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        """Returns the last predicted frame (3, H, W) float32 [0,1] on CPU."""
        return torch.from_numpy(self._rollout(history_images_chw, actions)[-1]).float()

    @torch.no_grad()
    def rollout_trajectory(
        self,
        history_images_chw: np.ndarray,  # (N >= 4, 3, H, W) float32 in [0,1]
        actions: np.ndarray,              # (T, 2)  float32 in env pixel coords [0,511]
    ) -> np.ndarray:
        """Returns all T predicted frames as (T, 3, H, W) float32 [0,1]."""
        return np.stack(self._rollout(history_images_chw, actions))
