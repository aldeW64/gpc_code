"""Adapter wrapping interactive_world_sim LatentWorldModel for GPC evaluation.

Provides the same rollout_final_image() interface as WamWorldModelAdapter and
GpcWorldModelAdapter so it can be used as a drop-in world-model backend in
eval_wam.py.

Loading protocol
----------------
The adapter needs both the Lightning checkpoint (.ckpt) and the Hydra config
(config.yaml) that was generated during training.  The config is auto-detected
from the checkpoint path using the standard Hydra output tree:

    <outputs_root>/<date>/<time>/.hydra/config.yaml
    <outputs_root>/<date>/<time>/checkpoints/<step>.ckpt   ← ckpt_path

If cfg_path is supplied explicitly it takes priority.

Batch interface (matches LatentWorldModel)
------------------------------------------
  history_images_chw : (N_hist, 3, H, W) float32 [0,1]
  actions            : (T_future, 2)     float32  pixel coords [0,511]

Returns a (3, H, W) float32 tensor in [0,1] — the predicted final frame.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf



def _auto_detect_cfg_path(ckpt_path: str) -> Optional[str]:
    """Walk up from the checkpoint to find .hydra/config.yaml."""
    p = Path(ckpt_path).resolve()
    # Typical Hydra layout: outputs/<date>/<time>/checkpoints/<name>.ckpt
    # .hydra sits two levels above the checkpoints/ dir.
    for ancestor in p.parents:
        candidate = ancestor / ".hydra" / "config.yaml"
        if candidate.exists():
            return str(candidate)
    return None


@dataclass
class IWSAdapterConfig:
    ckpt_path: str
    cfg_path: str = ""          # auto-detected from ckpt_path if empty
    num_history: int = 1        # number of context frames fed to the encoder
    num_frames: int = 8         # number of future frames to roll out


class IWSWorldModelAdapter:
    """
    Wraps a trained interactive_world_sim LatentWorldModel as a rollout oracle.

    Pipeline:
      1. Encode the last history frame(s) → latent(s) via model.encoder_forward()
      2. Roll out future latents via model.dynamics_forward()
      3. Decode the final latent → pixel image via render_img_cm()
    """

    def __init__(
        self,
        cfg: IWSAdapterConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Deferred: only add interactive_world_sim to sys.path and import its
        # modules here, so that importing iws_adapter itself (at module scope)
        # does not pull in the IWS package.
        _iws_root = Path(__file__).absolute().parents[1] / "interactive_world_sim"
        if str(_iws_root) not in sys.path:
            sys.path.insert(0, str(_iws_root))
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
        if not OmegaConf.has_resolver("torch"):
            OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
        from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
        from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
        self._render_img_cm = render_img_cm

        # --- resolve hydra config ---
        cfg_path = cfg.cfg_path if cfg.cfg_path else _auto_detect_cfg_path(cfg.ckpt_path)
        if cfg_path is None:
            raise FileNotFoundError(
                f"Cannot find .hydra/config.yaml for checkpoint {cfg.ckpt_path}. "
                "Set cfg_path explicitly in IWSAdapterConfig."
            )
        hydra_cfg = OmegaConf.load(cfg_path)
        algo_cfg = hydra_cfg.algorithm

        # --- load model ---
        self.model: LatentWorldModel = LatentWorldModel.load_from_checkpoint(
            cfg.ckpt_path,
            cfg=algo_cfg,
            map_location=self.device,
            weights_only=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Normalizer is saved inside the Lightning checkpoint (it's an nn.Module).
        # Verify it was actually fitted — if not, build a default PushT normalizer.
        try:
            _ = self.model.normalizer["action"]
        except (KeyError, RuntimeError):
            print(
                "[iws_adapter] WARNING: normalizer not found in checkpoint, "
                "building default PushT normalizer (action range [0,511]).",
                flush=True,
            )
            self.model.set_normalizer(self._default_pusht_normalizer())

        self.resolution: int = int(algo_cfg.diffusion.image_size)
        print(
            f"[iws_adapter] loaded IWS model from {cfg.ckpt_path} "
            f"(stage={algo_cfg.training_stage}, resolution={self.resolution})",
            flush=True,
        )

    @staticmethod
    def _default_pusht_normalizer():
        """Fallback normalizer using known PushT action range."""
        import numpy as np
        from interactive_world_sim.utils.normalizer import (
            LinearNormalizer,
            get_image_range_normalizer,
            get_range_normalizer_from_stat,
        )

        normalizer = LinearNormalizer()
        normalizer["image"] = get_image_range_normalizer()
        action_stat = {
            "min": np.array([0.0, 0.0], dtype=np.float32),
            "max": np.array([511.0, 511.0], dtype=np.float32),
            "mean": np.array([255.5, 255.5], dtype=np.float32),
            "std": np.array([147.6, 147.6], dtype=np.float32),
        }
        normalizer["action"] = get_range_normalizer_from_stat(action_stat)
        return normalizer

    @torch.no_grad()
    def rollout_final_image(
        self,
        history_images_chw: np.ndarray,   # (N_hist, 3, H, W) float32 [0,1]
        actions: np.ndarray,               # (T_future, 2) float32 pixel coords [0,511]
    ) -> torch.Tensor:
        """Predict the final frame after executing `actions` from the given history.

        Returns (3, H, W) float32 in [0,1] on CPU.
        """
        cfg = self.cfg
        N_hist = history_images_chw.shape[0]
        T_future = len(actions)
        assert N_hist >= 1, "Need at least one history frame"

        model = self.model
        device = self.device
        dtype = model.dtype

        # ---- normalise images and encode to latents ----
        # Use only the last cfg.num_history frames for the encoder.
        n_use = min(cfg.num_history, N_hist)
        hist_np = history_images_chw[-n_use:]  # (n_use, 3, H, W) float32 [0,1]

        hist_t = torch.from_numpy(hist_np).to(device=device, dtype=dtype)  # (n_use,3,H,W)
        # Normalise images to [-1,1]
        hist_norm = model.normalizer["image"].normalize(hist_t)  # (n_use,3,H,W)

        # encoder_forward expects (B, C, H, W)
        z_hist = model.encoder_forward(hist_norm)  # (n_use, C_lat, H_lat, W_lat)
        # shape: (1, n_use, C_lat, H_lat, W_lat)
        z_hist = z_hist.unsqueeze(0)

        # ---- normalise actions ----
        actions_t = torch.from_numpy(actions.astype(np.float32)).to(device=device, dtype=dtype)
        actions_norm = model.normalizer["action"].normalize(actions_t)  # (T_future, 2)

        # dynamics_forward signature:
        #   z_0     : (B, T_hist, C, H, W)
        #   action  : (B, T_hist + T_future, A)
        # We treat n_use history frames as the T_hist context and prepend
        # n_use zero-action slots for the history window.
        hist_actions = torch.zeros(1, n_use, actions_norm.shape[-1], device=device, dtype=dtype)
        future_actions = actions_norm.unsqueeze(0)  # (1, T_future, 2)
        all_actions = torch.cat([hist_actions, future_actions], dim=1)  # (1, n_use+T_future, 2)

        # Roll out future latents: (1, T_future, C_lat, H_lat, W_lat)
        z_future = model.dynamics_forward(z_hist, all_actions)

        # Take the last predicted latent and decode it.
        z_final = z_future[:, -1, :, :, :]  # (1, C_lat, H_lat, W_lat)

        # render_img_cm returns (B, 3*num_views, H, W) in [0,1]
        img = self._render_img_cm(
            model,
            z_final,
            self.resolution,
            model.normalizer,
            num_views=model.num_views,
        )  # (1, 3, H, W)

        return img[0].cpu().float()  # (3, H, W)
