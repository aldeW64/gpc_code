"""
visualize_predictions.py — Multimodal world model prediction visualizer.

Randomly samples trajectory windows from the ManiFEEL dataset, runs an
autoregressive world-model rollout, and saves a side-by-side comparison
video (ground truth vs. predicted) per episode.

The goal image is the last RGB frame of the sampled sequence window.

Usage (run from repo root):

    python world_model_multimodal/visualize_predictions.py \\
        --fusion_type early \\
        --ckpt_path world_model_multimodal/early_fusion/saved_checkpoints_phase2/checkpoint_epoch_2/denoiser.pth \\
        --dataset_path dataset/manifeel/data \\
        --num_episodes 5 \\
        --output_dir world_model_multimodal/viz_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
import zarr
import imageio

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_DIRS = [
    "nutbolt_quan_July1",
    "bulb_quan_Sep19",
    "gear_quan_Sep15",
    "pih_quan_June06",
    "plug_quan_Aug02",
    "usb_quan_Aug05",
    "blindinsert_quan_Aug15",
    "sorting_quan_Aug8",
    "explore_quan_June17",
]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _chw11_to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 in [-1, 1] → (H, W, 3) uint8."""
    img = np.clip((img + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return np.moveaxis(img, 0, -1)


def _chw01_to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 in [0, 1] → (H, W, 3) uint8."""
    return np.clip(np.moveaxis(img, 0, -1) * 255.0, 0, 255).astype(np.uint8)


def _resize_hwc(img_hwc: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize (H, W, 3) float32 → (size, size, 3) float32."""
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def _add_text(img_hwc: np.ndarray, text: str) -> np.ndarray:
    """
    Overlay `text` in the top-left corner of a (H, W, 3) uint8 image.
    Uses PIL when available; falls back to a plain coloured strip if not.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img_hwc)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/lib64/ncarg/database/ftfonts/VeraBd.ttf", size=14)
        except Exception:
            font = ImageFont.load_default()
        # Dark shadow for contrast on any background
        draw.text((6, 6), text, font=font, fill=(0, 0, 0))
        draw.text((5, 5), text, font=font, fill=(255, 255, 255))
        return np.array(pil)
    except ImportError:
        # PIL not available: draw a 16-px coloured banner at the top
        out = img_hwc.copy()
        out[:16, :] = (30, 30, 30)
        return out


def _build_video_frame(
    gt_rgb_hwc: np.ndarray,
    pr_rgb_hwc: np.ndarray,
    gt_tac_hwc: np.ndarray,
    pr_tac_hwc: np.ndarray,
    step_label: str = "",
) -> np.ndarray:
    """
    Tile four (H, W, 3) uint8 panels into a labelled 2×2 grid:

        GT Front       | Prediction Front
        GT Tactile     | Prediction Tactile

    A thin header strip above the grid shows column labels and an optional
    step label (e.g. "Context t=0" or "Pred step 3/8").
    """
    H, W = gt_rgb_hwc.shape[:2]

    # Per-panel captions (drawn in the top-left corner of each panel)
    gt_rgb_hwc  = _add_text(gt_rgb_hwc,  "GT Front")
    pr_rgb_hwc  = _add_text(pr_rgb_hwc,  "Pred Front")
    gt_tac_hwc  = _add_text(gt_tac_hwc,  "GT Tactile")
    pr_tac_hwc  = _add_text(pr_tac_hwc,  "Pred Tactile")

    top = np.concatenate([gt_rgb_hwc, pr_rgb_hwc], axis=1)   # (H, 2W, 3)
    bot = np.concatenate([gt_tac_hwc, pr_tac_hwc], axis=1)   # (H, 2W, 3)
    grid = np.concatenate([top, bot], axis=0)                 # (2H, 2W, 3)

    # Header bar with column titles + step label
    bar_h = 22
    bar = np.full((bar_h, 2 * W, 3), 30, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_bar = Image.fromarray(bar)
        draw = ImageDraw.Draw(pil_bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=13)
        except Exception:
            font = ImageFont.load_default()
        # Column labels centred over each half
        draw.text((W // 2 - 40, 4), "Ground Truth", font=font, fill=(180, 220, 180))
        draw.text((W + W // 2 - 38, 4), "Prediction",  font=font, fill=(220, 180, 180))
        if step_label:
            draw.text((2 * W - 120, 4), step_label, font=font, fill=(200, 200, 200))
        bar = np.array(pil_bar)
    except ImportError:
        pass  # use plain dark bar

    return np.concatenate([bar, grid], axis=0)   # (2H + bar_h, 2W, 3)


# ---------------------------------------------------------------------------
# Action normalisation helpers
# ---------------------------------------------------------------------------

def _compute_action_stats(action_raw: np.ndarray) -> dict:
    """Return {'min': (7,), 'max': (7,)} numpy float32 arrays."""
    return {
        "min": action_raw.min(axis=0).astype(np.float32),
        "max": action_raw.max(axis=0).astype(np.float32),
    }


def _normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    denom = stats["max"] - stats["min"]
    denom = np.where(denom < 1e-8, 1.0, denom)
    return ((action - stats["min"]) / denom * 2.0 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------

def _load_zarr_task(zarr_path: str, obs_horizon: int, pred_horizon: int, resize: int, rng: np.random.Generator):
    """
    Randomly sample one trajectory window from a zarr task store.

    Returns
    -------
    front   : (window_size, 3, H, H) float32 in [-1, 1]
    tactile : (window_size, 3, H, H) float32 in [-1, 1]
    action  : (window_size, 7) float32 in [-1, 1]
    action_stats : dict with 'min'/'max' arrays
    or None if no valid window found.
    """
    window_size = obs_horizon + pred_horizon
    store = zarr.open(zarr_path, mode="r")

    episode_ends = np.asarray(store["meta"]["episode_ends"][:], dtype=np.int64)
    episode_starts = np.zeros_like(episode_ends)
    episode_starts[1:] = episode_ends[:-1]

    # Collect valid (start, end) windows
    valid: list[tuple[int, int]] = []
    for s, e in zip(episode_starts, episode_ends):
        ep_len = int(e) - int(s)
        if ep_len >= window_size:
            for ws in range(int(s), int(e) - window_size + 1):
                valid.append((ws, ws + window_size))
    if not valid:
        return None

    ws, we = valid[int(rng.integers(0, len(valid)))]

    front_raw = store["data"]["front"][ws:we]           # (T, 256, 256, 3) float32
    tac_raw = store["data"]["left_tactile_camera_taxim"][ws:we]  # (T, 320, 240, 3) float32
    action_raw = store["data"]["action"][ws:we]         # (T, D) float32

    # Pad to 7-DOF if needed
    if action_raw.shape[1] < 7:
        action_raw = np.concatenate(
            [action_raw, np.zeros((len(action_raw), 7 - action_raw.shape[1]), dtype=np.float32)],
            axis=1,
        )

    full_action = store["data"]["action"][:].astype(np.float32)
    if full_action.shape[1] < 7:
        full_action = np.concatenate(
            [full_action, np.zeros((len(full_action), 7 - full_action.shape[1]), dtype=np.float32)],
            axis=1,
        )
    action_stats = _compute_action_stats(full_action)
    action_norm = _normalize_action(action_raw, action_stats)  # (T, 7) in [-1, 1]

    T = window_size
    front_out = np.zeros((T, 3, resize, resize), dtype=np.float32)
    tac_out = np.zeros((T, 3, resize, resize), dtype=np.float32)

    for t in range(T):
        f = _resize_hwc(front_raw[t].astype(np.float32), resize)   # (H, H, 3) in [0, 1]
        ta = _resize_hwc(tac_raw[t].astype(np.float32), resize)    # (H, H, 3) in [0, 1]
        # Clamp values; ManiFEEL stores are float32 already in [0, 1]
        f = np.clip(f, 0.0, 1.0)
        ta = np.clip(ta, 0.0, 1.0)
        front_out[t] = (f.transpose(2, 0, 1) * 2.0 - 1.0)   # [-1, 1]
        tac_out[t] = (ta.transpose(2, 0, 1) * 2.0 - 1.0)    # [-1, 1]

    return front_out, tac_out, action_norm, action_stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(fusion_type: str, ckpt_path: str, obs_horizon: int, device: torch.device):
    """
    Load the denoiser + sampler for the requested fusion type.
    Returns (sampler, fusion_type_str).
    """
    fusion_dir = os.path.join(_THIS_DIR, f"{fusion_type}_fusion")
    if fusion_dir not in sys.path:
        sys.path.insert(0, fusion_dir)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    if fusion_type == "early":
        from diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
        from diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
        from diffusion.inner_model import InnerModelConfig

        inner_cfg = InnerModelConfig(
            img_channels=6,
            num_steps_conditioning=obs_horizon,
            cond_channels=256,
            depths=[2, 2, 2, 2],
            channels=[96, 96, 96, 96],
            attn_depths=[False, False, True, True],
            num_actions=7,
            is_upsampler=False,
        )
        denoiser_cfg = DenoiserConfig(
            inner_model=inner_cfg,
            sigma_data=0.5,
            sigma_offset_noise=0.1,
            noise_previous_obs=True,
            upsampling_factor=None,
        )
        sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20.0)
        denoiser = Denoiser(denoiser_cfg)
        denoiser.setup_training(sigma_cfg)
        denoiser.load_state_dict(state)
        denoiser.to(device).eval()
        sampler = DiffusionSampler(denoiser, DiffusionSamplerConfig(num_steps_denoising=3))
        return sampler

    elif fusion_type == "middle":
        from diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
        from diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
        from diffusion.inner_model import MiddleFusionInnerModelConfig

        inner_cfg = MiddleFusionInnerModelConfig(
            img_channels=3,
            num_steps_conditioning=obs_horizon,
            num_actions=7,
            cond_channels=256,
            depths=[2, 2, 2, 2],
            channels=[96, 96, 96, 96],
            attn_depths=[0, 0, 1, 1],
        )
        denoiser_cfg = DenoiserConfig(
            inner_model=inner_cfg,
            sigma_data=0.5,
            sigma_offset_noise=0.1,
            noise_previous_obs=True,
        )
        sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20.0)
        denoiser = Denoiser(denoiser_cfg)
        denoiser.setup_training(sigma_cfg)
        denoiser.load_state_dict(state)
        denoiser.to(device).eval()
        sampler = DiffusionSampler(denoiser, DiffusionSamplerConfig(num_steps_denoising=3))
        return sampler

    elif fusion_type == "late":
        from diffusion.denoiser import LateFusionDenoiser, LateFusionDenoiserConfig, SigmaDistributionConfig
        from diffusion.diffusion_sampler import LateFusionDiffusionSampler, LateFusionDiffusionSamplerConfig

        denoiser_cfg = LateFusionDenoiserConfig(
            img_channels=3,
            num_steps_conditioning=obs_horizon,
            cond_channels=256,
            action_dim=7,
            depths=[2, 2, 2, 2],
            channels=[96, 96, 96, 96],
            attn_depths=[0, 0, 1, 1],
            sigma_data=0.5,
            sigma_offset_noise=0.1,
            noise_previous_obs=True,
        )
        sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20.0)
        denoiser = LateFusionDenoiser(denoiser_cfg)
        denoiser.setup_training(sigma_cfg)
        denoiser.load_state_dict(state)
        denoiser.to(device).eval()
        sampler = LateFusionDiffusionSampler(denoiser, LateFusionDiffusionSamplerConfig(num_steps_denoising=3))
        return sampler

    else:
        raise ValueError(f"Unknown fusion_type={fusion_type!r}. Expected early|middle|late.")


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_early(
    sampler,
    front: np.ndarray,   # (window_size, 3, H, W) in [-1,1]
    tactile: np.ndarray, # (window_size, 3, H, W) in [-1,1]
    action: np.ndarray,  # (window_size, 7) in [-1,1]
    obs_horizon: int,
    pred_horizon: int,
    device: torch.device,
) -> list[tuple]:
    """
    Returns list of pred_horizon tuples:
        (gt_rgb_hwc, pred_rgb_hwc, gt_tac_hwc, pred_tac_hwc) all uint8
    """
    # Fused frames: (T, 6, H, W)
    fused = np.concatenate([front, tactile], axis=1)  # (T, 6, H, W)
    buffer = list(fused[:obs_horizon])  # obs_horizon × (6, H, W)

    results = []
    for t in range(pred_horizon):
        # Context window
        prev_obs = np.stack(buffer[-obs_horizon:], axis=0)   # (n, 6, H, W)
        prev_act = action[t : t + obs_horizon]                # (n, 7)

        prev_obs_t = torch.from_numpy(prev_obs).unsqueeze(0).to(device)  # (1, n, 6, H, W)
        prev_act_t = torch.from_numpy(prev_act).unsqueeze(0).to(device)  # (1, n, 7)

        pred, _ = sampler.sample(prev_obs_t, prev_act_t)   # (1, 6, H, W) in [-1, 1]
        pred_np = pred.squeeze(0).cpu().float().numpy()    # (6, H, W)

        buffer.append(pred_np)

        # Ground-truth next frame
        gt_frame = fused[obs_horizon + t]  # (6, H, W)

        results.append((
            _chw11_to_hwc_uint8(gt_frame[:3]),    # gt front
            _chw11_to_hwc_uint8(pred_np[:3]),     # pred front
            _chw11_to_hwc_uint8(gt_frame[3:]),    # gt tactile
            _chw11_to_hwc_uint8(pred_np[3:]),     # pred tactile
        ))

    return results


@torch.no_grad()
def _rollout_dual_stream(
    sampler,
    front: np.ndarray,   # (window_size, 3, H, W) in [-1,1]
    tactile: np.ndarray, # (window_size, 3, H, W) in [-1,1]
    action: np.ndarray,  # (window_size, 7) in [-1,1]
    obs_horizon: int,
    pred_horizon: int,
    device: torch.device,
) -> list[tuple]:
    """
    Rollout for middle and late fusion (both use the same sampler signature).
    Returns list of pred_horizon tuples:
        (gt_rgb_hwc, pred_rgb_hwc, gt_tac_hwc, pred_tac_hwc) all uint8
    """
    rgb_buffer = list(front[:obs_horizon])     # obs_horizon × (3, H, W)
    tac_buffer = list(tactile[:obs_horizon])   # obs_horizon × (3, H, W)

    results = []
    for t in range(pred_horizon):
        prev_rgb = np.stack(rgb_buffer[-obs_horizon:], axis=0)   # (n, 3, H, W)
        prev_tac = np.stack(tac_buffer[-obs_horizon:], axis=0)   # (n, 3, H, W)
        prev_act = action[t : t + obs_horizon]                    # (n, 7)

        prev_rgb_t = torch.from_numpy(prev_rgb).unsqueeze(0).to(device)  # (1, n, 3, H, W)
        prev_tac_t = torch.from_numpy(prev_tac).unsqueeze(0).to(device)
        prev_act_t = torch.from_numpy(prev_act).unsqueeze(0).to(device)

        (pred_rgb, pred_tac), _ = sampler.sample(prev_rgb_t, prev_tac_t, prev_act_t)

        pred_rgb_np = pred_rgb.squeeze(0).cpu().float().numpy()  # (3, H, W) in [-1, 1]
        pred_tac_np = pred_tac.squeeze(0).cpu().float().numpy()

        rgb_buffer.append(pred_rgb_np)
        tac_buffer.append(pred_tac_np)

        gt_frame_rgb = front[obs_horizon + t]    # (3, H, W)
        gt_frame_tac = tactile[obs_horizon + t]  # (3, H, W)

        results.append((
            _chw11_to_hwc_uint8(gt_frame_rgb),
            _chw11_to_hwc_uint8(pred_rgb_np),
            _chw11_to_hwc_uint8(gt_frame_tac),
            _chw11_to_hwc_uint8(pred_tac_np),
        ))

    return results


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def _save_video(path: str, frames: list, fps: int = 5) -> None:
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=None,
                                macro_block_size=None, pixelformat="yuv420p")
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def _save_goal_image(path: str, img_chw11: np.ndarray) -> None:
    """Save (3, H, W) float32 in [-1, 1] as PNG."""
    imageio.imwrite(path, _chw11_to_hwc_uint8(img_chw11))


def _save_context_image(path: str, img_chw11: np.ndarray) -> None:
    """Save (3, H, W) float32 in [-1, 1] as PNG."""
    imageio.imwrite(path, _chw11_to_hwc_uint8(img_chw11))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize multimodal world model predictions vs. GT.")
    parser.add_argument("--fusion_type", choices=["early", "middle", "late"], default="early")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to denoiser.pth checkpoint")
    parser.add_argument("--dataset_path", type=str,
                        default="/n/holylabs/ydu_lab/Lab/pwu/Projects/gpc_code/dataset/manifeel/data")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--obs_horizon", type=int, default=4)
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument("--resize_scale", type=int, default=96)
    parser.add_argument("--output_dir", type=str, default="world_model_multimodal/viz_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build list of available task zarr paths
    task_paths = [
        os.path.join(args.dataset_path, t)
        for t in TASK_DIRS
        if os.path.isdir(os.path.join(args.dataset_path, t))
    ]
    if not task_paths:
        raise FileNotFoundError(f"No task directories found in {args.dataset_path}")
    print(f"Found {len(task_paths)} tasks: {[os.path.basename(p) for p in task_paths]}")

    # Load model
    print(f"Loading {args.fusion_type} fusion model from {args.ckpt_path} …")
    sampler = _load_model(args.fusion_type, args.ckpt_path, args.obs_horizon, device)
    print("Model loaded.")

    ep = 0
    attempts = 0
    max_attempts = args.num_episodes * 20

    while ep < args.num_episodes and attempts < max_attempts:
        attempts += 1
        zarr_path = task_paths[int(rng.integers(0, len(task_paths)))]
        task_name = os.path.basename(zarr_path)

        result = _load_zarr_task(
            zarr_path=zarr_path,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            resize=args.resize_scale,
            rng=rng,
        )
        if result is None:
            continue

        front, tactile, action, action_stats = result
        # front, tactile: (window_size, 3, H, W) in [-1, 1]
        # action: (window_size, 7) in [-1, 1]

        # Goal image = last RGB frame in the prediction window
        goal_rgb = front[args.obs_horizon + args.pred_horizon - 1]   # (3, H, W) in [-1, 1]
        context_rgb = front[args.obs_horizon - 1]                     # last context frame

        print(f"[ep {ep+1}/{args.num_episodes}] task={task_name}", flush=True)

        # Autoregressive rollout
        if args.fusion_type == "early":
            step_results = _rollout_early(
                sampler, front, tactile, action,
                args.obs_horizon, args.pred_horizon, device,
            )
        else:
            step_results = _rollout_dual_stream(
                sampler, front, tactile, action,
                args.obs_horizon, args.pred_horizon, device,
            )

        # Build full video: context frames (GT only) + prediction frames (GT vs pred)
        video_frames = []

        # Context frames: show GT on both sides (nothing to predict yet)
        for t in range(args.obs_horizon):
            gt_rgb = _chw11_to_hwc_uint8(front[t])
            gt_tac = _chw11_to_hwc_uint8(tactile[t])
            frame = _build_video_frame(
                gt_rgb, gt_rgb, gt_tac, gt_tac,
                step_label=f"Context {t+1}/{args.obs_horizon}",
            )
            video_frames.append(frame)

        # Prediction frames: GT vs model prediction
        for i, (gt_rgb, pred_rgb, gt_tac, pred_tac) in enumerate(step_results):
            frame = _build_video_frame(
                gt_rgb, pred_rgb, gt_tac, pred_tac,
                step_label=f"Step {i+1}/{args.pred_horizon}",
            )
            video_frames.append(frame)

        # Save outputs
        prefix = f"ep{ep:03d}_{task_name}_{args.fusion_type}"
        vid_path = os.path.join(args.output_dir, f"{prefix}.mp4")
        goal_path = os.path.join(args.output_dir, f"{prefix}_goal.png")
        ctx_path = os.path.join(args.output_dir, f"{prefix}_context.png")

        _save_video(vid_path, video_frames, fps=args.fps)
        _save_goal_image(goal_path, goal_rgb)
        _save_context_image(ctx_path, context_rgb)

        print(f"  Saved {vid_path}")
        print(f"  Goal image (last frame of sequence): {goal_path}")
        print(f"  Context image (last input frame):    {ctx_path}")

        ep += 1

    print(f"\nDone. {ep} episodes saved to {args.output_dir}")
    print("Video layout: top-left=GT front, top-right=Pred front,")
    print("              bottom-left=GT tactile, bottom-right=Pred tactile.")
    print("First obs_horizon frames show GT context (both sides identical).")


if __name__ == "__main__":
    main()
