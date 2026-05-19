from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'torch'. Please activate the project environment (see environment.yml) "
        "before running gpc_wam_evaluation."
    ) from e

# Force imports to resolve from this local repo, not a globally installed package.
_REPO_ROOT = Path(__file__).absolute().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_WAM_PACKAGE_ROOT = _REPO_ROOT / "wam"
if str(_WAM_PACKAGE_ROOT) not in sys.path:
    # Needed because local wam code uses absolute imports like `from models...`.
    sys.path.insert(0, str(_WAM_PACKAGE_ROOT))

from wam.config import wm_args
from wam.models.ctrl_world import CrtlWorld
from wam.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

# Safety check: ensure `wam` is loaded from this repo.
_WAM_ROOT = (_REPO_ROOT / "wam").resolve()
_WAM_CONFIG_FILE = Path(sys.modules["wam.config"].__file__).resolve()
if _WAM_ROOT not in _WAM_CONFIG_FILE.parents:
    raise ImportError(
        f"`wam` resolved outside local repo: {_WAM_CONFIG_FILE}. "
        f"Expected under {_WAM_ROOT}. Please run from repo root and avoid external `wam` packages."
    )


@dataclass
class WamAdapterConfig:
    ckpt_path: str
    svd_model_path: str
    clip_model_path: str

    num_history: int
    num_frames: int
    action_dim: int
    width: int
    height: int
    fps: int
    motion_bucket_id: int
    guidance_scale: float
    num_inference_steps: int
    decode_chunk_size: int
    dtype: str = "bf16"  # bf16 | fp16 | fp32


class WamWorldModelAdapter:
    """
    Minimal adapter that treats Ctrl-World as a rollout model producing future frames
    from (history_latents + action sequence).

    Push-T is single-view; Ctrl-World expects 3-view layout in height dimension.
    We tile the single latent into 3 views to satisfy the pipeline layout.
    """

    def __init__(self, cfg: WamAdapterConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[cfg.dtype]

        args = wm_args()
        args.svd_model_path = cfg.svd_model_path
        args.clip_model_path = cfg.clip_model_path
        args.ckpt_path = cfg.ckpt_path
        args.val_model_path = cfg.ckpt_path

        args.num_history = cfg.num_history
        args.num_frames = cfg.num_frames
        args.action_dim = cfg.action_dim
        args.width = cfg.width
        args.height = cfg.height
        args.fps = cfg.fps
        args.motion_bucket_id = cfg.motion_bucket_id
        args.guidance_scale = cfg.guidance_scale
        args.num_inference_steps = cfg.num_inference_steps
        args.decode_chunk_size = cfg.decode_chunk_size
        args.frame_level_cond = True
        args.text_cond = False
        args.his_cond_zero = False

        self.args = args

        self.model = CrtlWorld(args)
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        # Replace the plain SVD pipeline with a CtrlWorldDiffusionPipeline so that
        # __call__ is dispatched correctly via normal method resolution rather than
        # as an unbound method on a StableVideoDiffusionPipeline instance.
        ctrl_pipeline = CtrlWorldDiffusionPipeline(**self.model.pipeline.components)
        ctrl_pipeline.to(self.device)
        self.model.pipeline = ctrl_pipeline
        self.pipeline = ctrl_pipeline

    def _resize_chw(self, image_chw: torch.Tensor) -> torch.Tensor:
        """
        image_chw: (3,H,W) float in [0,1]
        returns: (3,cfg.height,cfg.width)
        """
        target_h = int(self.cfg.height)
        target_w = int(self.cfg.width)
        if image_chw.shape[-2:] == (target_h, target_w):
            return image_chw
        x = image_chw.unsqueeze(0)  # (1,3,H,W)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    @torch.no_grad()
    def encode_image_to_latent(self, image_chw: np.ndarray) -> torch.Tensor:
        """
        image_chw: (3,H,W) float32 in [0,1]
        Returns latent (1,4,32,32) on device/dtype.
        """
        img = torch.from_numpy(image_chw).to(self.device, dtype=self.dtype)
        img = self._resize_chw(img)
        img = img.unsqueeze(0)  # (1,3,H,W)
        img = img * 2.0 - 1.0
        vae = self.pipeline.vae
        latent = vae.encode(img).latent_dist.sample().mul_(vae.config.scaling_factor)
        return latent

    @torch.no_grad()
    def decode_latent_to_image(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B,4,32,32)
        Returns (B,3,H,W) float32 in [0,1]
        """
        vae = self.pipeline.vae
        x = latent / vae.config.scaling_factor
        decoded = vae.decode(x, num_frames=x.shape[0]).sample
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1).to(torch.float32)
        return decoded

    def _tile_to_three_views(self, latent: torch.Tensor) -> torch.Tensor:
        # latent (1,4,h,w) -> (1,4,3*h,w) by tiling in height
        return torch.cat([latent, latent, latent], dim=2)

    @torch.no_grad()
    def rollout_final_image(
        self,
        history_images_chw: np.ndarray,
        actions_t2: np.ndarray,
    ) -> torch.Tensor:
        """
        history_images_chw: (Hh, 3, H, W) float32 in [0,1] where Hh==num_history
        actions_t2: (Tf, 2) in env coordinates [0,511]

        Returns predicted final image tensor (3,H,W) float32 in [0,1] on CPU.
        """
        cfg = self.cfg
        assert history_images_chw.shape[0] == cfg.num_history
        assert actions_t2.shape[0] == cfg.num_frames

        # encode history to latents and build history tensor (1, num_history, 4, 3*h_lat, w_lat)
        his_latents = []
        h_lat = None
        w_lat = None
        for i in range(cfg.num_history):
            lat = self.encode_image_to_latent(history_images_chw[i])
            # lat is (1,4,h_lat,w_lat)
            if h_lat is None:
                h_lat, w_lat = int(lat.shape[-2]), int(lat.shape[-1])
            lat = self._tile_to_three_views(lat)  # (1,4,96,32)
            his_latents.append(lat)
        history = torch.cat(his_latents, dim=0).unsqueeze(0)  # (1, Hh, 4, 96, 32)

        # Keep batch dim for pipeline input: (B,4,3*h_lat,w_lat), B=1.
        current_latent = his_latents[-1]  # (1,4,96,32)

        # Build WAM action condition: (num_history+num_frames, 7)
        # Push-T action is 2D; we pack into first 2 dims and zero-pad others.
        action_cond = np.zeros((cfg.num_history + cfg.num_frames, cfg.action_dim), dtype=np.float32)
        # Use last known agent position proxy for history actions (zeros), future use provided actions
        action_cond[cfg.num_history :, 0:2] = actions_t2.astype(np.float32)

        action_cond_t = torch.from_numpy(action_cond).unsqueeze(0).to(self.device, dtype=self.dtype)
        text_token = self.model.action_encoder(action_cond_t, frame_level_cond=True)

        _, pred_latents = self.pipeline(
            image=current_latent,
            text=text_token,
            width=cfg.width,
            height=int(cfg.height * 3),
            num_frames=cfg.num_frames,
            history=history,
            num_inference_steps=cfg.num_inference_steps,
            decode_chunk_size=cfg.decode_chunk_size,
            max_guidance_scale=cfg.guidance_scale,
            fps=cfg.fps,
            motion_bucket_id=cfg.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
            his_cond_zero=False,
        )

        # pred_latents: (1, num_frames, 4, 3*h_lat, w_lat). Take final, crop to one view.
        if h_lat is None or w_lat is None:
            raise RuntimeError("Failed to infer latent spatial size from history encodes.")
        final_lat = pred_latents[0, -1]  # (4,3*h_lat,w_lat)
        final_lat_one = final_lat[:, 0:h_lat, :]  # (4,h_lat,w_lat)
        img = self.decode_latent_to_image(final_lat_one.unsqueeze(0))[0].cpu()  # (3,H,W)
        return img

