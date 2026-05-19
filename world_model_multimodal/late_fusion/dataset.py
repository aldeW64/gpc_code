"""
ManifEEL multimodal dataset for the Late Fusion world model.

Loads RGB (front camera) and tactile (left_tactile_camera_taxim) frames together
with 7D actions from one manifeel zarr store.  All images are resized to
(resize_scale x resize_scale) and normalised to [-1, 1].  Actions are normalised
to [-1, 1] using per-dataset min/max statistics.

Zarr layout expected:
  <root>/
    data/
      front                     : (N, 256, 256, 3)  float32  HWC  [0, 1]
      left_tactile_camera_taxim : (N, 320, 240, 3)  float32  HWC  [0, 1]
      action                    : (N, 7)             float32
    meta/
      episode_ends              : (num_episodes,)    int64    exclusive end indices

Dataset items:
  'front'   : (obs_horizon + pred_horizon, 3, resize_scale, resize_scale)  float32  [-1,1]
  'tactile' : (obs_horizon + pred_horizon, 3, resize_scale, resize_scale)  float32  [-1,1]
  'action'  : (obs_horizon + pred_horizon, 7)                              float32  [-1,1]
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_hwc_to_tensor(img_hwc: np.ndarray, size: int) -> Tensor:
    """
    Convert a single HWC float32 [0,1] image to a CHW float32 [-1,1] tensor
    resized to (size x size).

    Args:
        img_hwc: (H, W, 3) float32 in [0, 1]
        size:    target spatial resolution

    Returns:
        (3, size, size) float32 in [-1, 1]
    """
    # (H, W, 3) → (3, H, W), keep as float
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).float()   # (3, H, W) [0, 1]
    # Add batch dim for interpolate: (1, 3, H, W)
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
    t = t.squeeze(0)  # (3, size, size)
    # Normalise to [-1, 1]
    t = t * 2.0 - 1.0
    return t


def _build_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
) -> List[Tuple[int, int]]:
    """
    Build a list of (start_frame, end_frame) pairs — one per valid sliding window.

    Only includes windows that fit entirely within one episode.
    Episodes shorter than sequence_length are skipped.

    Args:
        episode_ends:    exclusive end index of each episode, shape (E,)
        sequence_length: total window length = obs_horizon + pred_horizon

    Returns:
        list of (start, end) index pairs into the flat array
    """
    indices: List[Tuple[int, int]] = []
    prev_end = 0
    for end in episode_ends:
        episode_len = end - prev_end
        if episode_len >= sequence_length:
            for start in range(prev_end, end - sequence_length + 1):
                indices.append((start, start + sequence_length))
        prev_end = end
    return indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManifEELDataset(Dataset):
    """
    Sliding-window dataset over one manifeel zarr store.

    Args:
        zarr_path:   path to the zarr store directory
        obs_horizon: number of past frames used as conditioning context
        pred_horizon: number of future frames to predict (autoregressive steps)
        resize_scale: spatial resolution for both RGB and tactile frames
        action_stats: optional dict with 'min' and 'max' arrays of shape (7,)
                      for action normalisation.  If None, computed from the data.
    """

    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int,
        pred_horizon: int,
        resize_scale: int = 96,
        action_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_length = obs_horizon + pred_horizon
        self.resize_scale = resize_scale

        root = zarr.open(zarr_path, mode='r')
        data = root['data']
        meta = root['meta']

        # Read array metadata only (lazy load)
        self._front_arr = data['front']              # (N, 256, 256, 3) float32
        self._tac_arr = data['left_tactile_camera_taxim']  # (N, 320, 240, 3) float32
        action_raw = data['action'][:]               # (N, 7) float32  — load fully

        episode_ends = meta['episode_ends'][:]       # (E,) int64

        # Build action normalisation statistics
        if action_stats is None:
            amax = action_raw.max(axis=0)
            amin = action_raw.min(axis=0)
            self.action_stats = {'min': amin, 'max': amax}
        else:
            self.action_stats = action_stats

        # Normalise actions to [-1, 1]
        amin = self.action_stats['min']
        amax = self.action_stats['max']
        self._action_norm = (action_raw - amin) / (amax - amin + 1e-8) * 2.0 - 1.0

        # Build sliding-window index
        self._indices = _build_sample_indices(episode_ends, self.sequence_length)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        start, end = self._indices[idx]

        # Load raw frames from zarr (HWC float32 [0,1])
        front_raw = self._front_arr[start:end]   # (T, 256, 256, 3)
        tac_raw = self._tac_arr[start:end]        # (T, 320, 240, 3)

        # Resize and normalise each frame
        T = end - start
        front_tensors = torch.stack(
            [_resize_hwc_to_tensor(front_raw[t], self.resize_scale) for t in range(T)],
            dim=0,
        )   # (T, 3, S, S)

        tac_tensors = torch.stack(
            [_resize_hwc_to_tensor(tac_raw[t], self.resize_scale) for t in range(T)],
            dim=0,
        )   # (T, 3, S, S)

        # Actions (already normalised)
        action = torch.from_numpy(self._action_norm[start:end]).float()  # (T, 7)

        return {
            'front': front_tensors,    # (T, 3, S, S)  float32  [-1, 1]
            'tactile': tac_tensors,    # (T, 3, S, S)  float32  [-1, 1]
            'action': action,          # (T, 7)         float32  [-1, 1]
        }


# ---------------------------------------------------------------------------
# Multi-store dataset builder
# ---------------------------------------------------------------------------

MANIFEEL_TASK_DIRS = [
    'nutbolt_quan_July1',
    'bulb_quan_Sep19',
    'gear_quan_Sep15',
    'pih_quan_June06',
    'plug_quan_Aug02',
    'usb_quan_Aug05',
    'blindinsert_quan_Aug15',
    'sorting_quan_Aug8',
    'explore_quan_June17',
]


def build_combined_dataset(
    dataset_root: str,
    obs_horizon: int,
    pred_horizon: int,
    resize_scale: int = 96,
) -> torch.utils.data.ConcatDataset:
    """
    Build a ConcatDataset over all manifeel task zarr stores found under dataset_root.

    Actions are normalised per dataset (not globally), which is the correct approach
    when different tasks have very different action ranges.

    Args:
        dataset_root: path to the manifeel data root (contains task directories)
        obs_horizon:  number of past frames per window
        pred_horizon: number of future frames per window
        resize_scale: spatial resolution

    Returns:
        ConcatDataset of ManifEELDataset instances (one per task found)
    """
    datasets = []
    data_dir = os.path.join(dataset_root, 'data')

    for task in MANIFEEL_TASK_DIRS:
        zarr_path = os.path.join(data_dir, task)
        if not os.path.isdir(zarr_path):
            print(f"[dataset] Skipping missing task: {zarr_path}")
            continue
        ds = ManifEELDataset(
            zarr_path=zarr_path,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            resize_scale=resize_scale,
        )
        print(f"[dataset] {task}: {len(ds)} windows")
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"No datasets found under {data_dir}")

    return torch.utils.data.ConcatDataset(datasets)
