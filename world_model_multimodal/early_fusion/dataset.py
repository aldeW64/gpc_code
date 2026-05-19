"""
ManifEEL multimodal dataset for the Early Fusion world model.

Reads from zarr stores produced by the ManifEEL robot manipulation benchmark.
Each store contains:
    front                        : (N, 256, 256, 3) float32  — front RGB camera
    left_tactile_camera_taxim    : (N, 320, 240, 3) float32  — left tactile camera
    action                       : (N, 7) float32            — 7-DoF robot actions
    state                        : (N, 7) float32            — 7-DoF robot state
    meta/episode_ends            : (num_episodes,) int64     — episode boundary indices

A sample of length `window_size = obs_horizon + pred_horizon` is drawn from
each valid sliding window within every episode.

Returned dict per sample:
    'front'   : (window_size, 3, resize, resize)  float32  in [-1, 1]
    'tactile' : (window_size, 3, resize, resize)  float32  in [-1, 1]
    'action'  : (window_size, 7)                  float32  in [-1, 1]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import zarr


# ---------------------------------------------------------------------------
# Helper: per-array statistics (min / max along sample axis)
# ---------------------------------------------------------------------------

def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-feature min/max for a (N, D) array."""
    flat = data.reshape(-1, data.shape[-1])
    return {
        "min": np.min(flat, axis=0).astype(np.float32),
        "max": np.max(flat, axis=0).astype(np.float32),
    }


def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Map data from [min, max] -> [-1, 1]."""
    denom = stats["max"] - stats["min"]
    # Avoid division by zero for constant dimensions
    denom = np.where(denom < 1e-8, 1.0, denom)
    ndata = (data - stats["min"]) / denom  # [0, 1]
    return (ndata * 2 - 1).astype(np.float32)  # [-1, 1]


def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of normalize_data."""
    x = (ndata + 1) / 2  # [0, 1]
    return x * (stats["max"] - stats["min"]) + stats["min"]


# ---------------------------------------------------------------------------
# Helper: sliding window indices over episodes
# ---------------------------------------------------------------------------

def create_sample_indices(
    episode_ends: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Build a list of (buf_start, buf_end) pairs — one per valid sliding window.

    episode_ends[i] is the *exclusive* end index of episode i.
    Episode i spans [episode_ends[i-1], episode_ends[i]) with episode_ends[-1] = 0
    for i = 0.

    Only windows that fit fully within an episode are included (no padding).
    """
    indices: List[Tuple[int, int]] = []
    prev_end = 0
    for end in episode_ends:
        ep_len = end - prev_end
        if ep_len >= window_size:
            for start in range(prev_end, end - window_size + 1):
                indices.append((start, start + window_size))
        prev_end = int(end)
    return np.array(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_image_np(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize a single HWC image (H, W, 3) to (size, size, 3) using bilinear
    interpolation via torch, then return as numpy float32.

    Input dtype can be uint8 or float32.  Output is float32 in [0, 1] for
    uint8 input, or unchanged range for float32 input (values clipped to
    [0, 255] are assumed to be raw float images from the zarr store).
    """
    # Convert to float32 tensor in [0, 1]
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    img_f = np.clip(img_f, 0.0, 1.0)

    # CHW for torch
    t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()  # (size, size, 3)


def to_normalized_chw(img_hwc: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) float32 in [0, 1] -> (3, H, W) float32 in [-1, 1].
    """
    chw = img_hwc.transpose(2, 0, 1)  # (3, H, W)
    return (chw * 2.0 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-dataset class
# ---------------------------------------------------------------------------

class ManifEELEpisodeDataset(Dataset):
    """
    Loads one ManifEEL zarr store and exposes sliding-window samples.

    Parameters
    ----------
    zarr_path    : str   Path to the zarr store directory.
    obs_horizon  : int   Number of past frames used as context by the model.
    pred_horizon : int   Number of future frames to predict.
    resize       : int   Target spatial resolution for both camera streams (default 96).
    action_stats : dict  Optional pre-computed {'min', 'max'} for actions.
                         If None, stats are computed from this dataset's actions.
    """

    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int,
        pred_horizon: int,
        resize: int = 96,
        action_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.zarr_path = zarr_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.window_size = obs_horizon + pred_horizon
        self.resize = resize

        store = zarr.open(zarr_path, mode="r")

        # ---- episode boundaries ----
        episode_ends = store["meta"]["episode_ends"][:]  # (num_episodes,) int64

        # ---- actions ----
        action_raw = store["data"]["action"][:]  # (N, 7) float32

        # Compute or use provided action normalization statistics
        if action_stats is None:
            self.action_stats = get_data_stats(action_raw)
        else:
            self.action_stats = action_stats

        self.action_norm = normalize_data(action_raw, self.action_stats)  # (N, 7) in [-1, 1]

        # Load into memory to avoid zarr re-open issues with DataLoader workers.
        self._front = store["data"]["front"][:]      # (N, 256, 256, 3) float32
        self._tactile = store["data"]["left_tactile_camera_taxim"][:]  # (N, 320, 240, 3) float32

        # ---- sliding window indices ----
        self.indices = create_sample_indices(episode_ends, self.window_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buf_start, buf_end = self.indices[idx]

        # Slice raw images: (window_size, H, W, 3)
        front_raw = self._front[buf_start:buf_end]      # (T, 256, 256, 3)
        tactile_raw = self._tactile[buf_start:buf_end]  # (T, 320, 240, 3)
        action_window = self.action_norm[buf_start:buf_end]  # (T, 7)

        T = self.window_size
        front_out = np.zeros((T, 3, self.resize, self.resize), dtype=np.float32)
        tactile_out = np.zeros((T, 3, self.resize, self.resize), dtype=np.float32)

        for t in range(T):
            front_resized = resize_image_np(front_raw[t], self.resize)   # (96,96,3) in [0,1]
            tactile_resized = resize_image_np(tactile_raw[t], self.resize)  # (96,96,3) in [0,1]
            front_out[t] = to_normalized_chw(front_resized)    # (3,96,96) in [-1,1]
            tactile_out[t] = to_normalized_chw(tactile_resized)  # (3,96,96) in [-1,1]

        return {
            "front": torch.from_numpy(front_out),        # (T, 3, 96, 96)
            "tactile": torch.from_numpy(tactile_out),    # (T, 3, 96, 96)
            "action": torch.from_numpy(action_window),   # (T, 7)
        }


# ---------------------------------------------------------------------------
# Multi-dataset factory
# ---------------------------------------------------------------------------

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


def build_combined_dataset(
    dataset_path: str,
    obs_horizon: int,
    pred_horizon: int,
    resize: int = 96,
    shared_action_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[ConcatDataset, Dict[str, np.ndarray]]:
    """
    Build a ConcatDataset over all ManifEEL task zarr stores found in
    `dataset_path`.

    If `shared_action_stats` is provided it is used for all sub-datasets
    (recommended so action normalization is consistent across tasks).
    Otherwise stats are computed from the first dataset found and reused.

    Returns
    -------
    combined_dataset    : ConcatDataset
    action_stats        : dict with 'min'/'max' arrays (shape (7,))
    """
    datasets: List[ManifEELEpisodeDataset] = []
    action_stats = shared_action_stats

    # Find available task directories (allows for subsets)
    available = []
    for name in TASK_DIRS:
        p = os.path.join(dataset_path, name)
        if os.path.isdir(p):
            available.append(p)

    if not available:
        raise FileNotFoundError(
            f"No ManifEEL task directories found in '{dataset_path}'. "
            f"Expected subdirectories: {TASK_DIRS}"
        )

    # If no shared stats provided, compute from the first available store
    if action_stats is None:
        store0 = zarr.open(available[0], mode="r")
        action_stats = get_data_stats(store0["data"]["action"][:])
        del store0

    for zarr_path in available:
        ds = ManifEELEpisodeDataset(
            zarr_path=zarr_path,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            resize=resize,
            action_stats=action_stats,
        )
        datasets.append(ds)
        print(f"  Loaded '{os.path.basename(zarr_path)}': {len(ds)} windows")

    combined = ConcatDataset(datasets)
    print(f"Combined dataset: {len(combined)} total windows from {len(datasets)} tasks")
    return combined, action_stats
