"""
dataset.py — PyTorch Dataset for the ManiFeel multi-task manipulation data.

Each task is stored as a zarr directory with the following arrays:
  data/front                     : [N, 256, 256, 3]  float32, values in [0, 1]
  data/left_tactile_camera_taxim : [N, 320, 240, 3]  float32, values in [0, 1]
  data/action                    : [N, 7]             float32
  meta/episode_ends              : [num_episodes]     int64, exclusive end indices

Each dataset item is a sliding window of length (obs_horizon + pred_horizon + 1)
where:
  - obs_horizon frames are used as context (past observations)
  - pred_horizon frames are prediction targets (autoregressive steps)
  - The +1 accounts for the extra target frame beyond the prediction window

Returned dict keys:
  'front'   : (T, 3, H, W)  float32, normalised to [-1, 1]
  'tactile' : (T, 3, H, W)  float32, normalised to [-1, 1]
  'action'  : (T, 7)        float32, normalised to [-1, 1] per dataset min/max

where T = obs_horizon + pred_horizon + 1.

Both front and tactile images are resized to (resize_scale, resize_scale) using
bilinear interpolation.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize a (H, W, 3) float32 image to (size, size, 3) using bilinear
    interpolation via PyTorch.

    Args:
        img: numpy array of shape (H, W, 3), values in [0, 1]
        size: target spatial size

    Returns:
        numpy array of shape (size, size, 3)
    """
    # (H, W, 3) → (1, 3, H, W)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def _compute_action_stats(zarr_root: zarr.Group, expected_dim: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-dimension min and max of the action array, padded to expected_dim."""
    action = zarr_root["data"]["action"][:]
    if action.shape[1] < expected_dim:
        pad = np.zeros((len(action), expected_dim - action.shape[1]), dtype=np.float32)
        action = np.concatenate([action, pad], axis=1)
    return action.min(axis=0), action.max(axis=0)


def _normalize_action(
    action: np.ndarray,
    act_min: np.ndarray,
    act_max: np.ndarray,
) -> np.ndarray:
    """Linearly normalise action to [-1, 1] using pre-computed min/max."""
    rng = act_max - act_min
    rng = np.where(rng < 1e-8, 1.0, rng)  # avoid division by zero
    return (action - act_min) / rng * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Per-task zarr dataset
# ---------------------------------------------------------------------------


class ManiFEELZarrTaskDataset(Dataset):
    """
    Dataset for a single ManiFEEL task zarr store.

    Args:
        zarr_path   : path to the zarr directory for one task
        obs_horizon : number of past frames used as context
        pred_horizon: number of future frames to predict
        resize_scale: target spatial resolution (both H and W)
        act_min/max : optional pre-computed action statistics for normalisation;
                      if None they are computed from this task's data
    """

    def __init__(
        self,
        zarr_path: str,
        obs_horizon: int,
        pred_horizon: int,
        resize_scale: int = 96,
        act_min: Optional[np.ndarray] = None,
        act_max: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.resize_scale = resize_scale

        # Sliding window length: obs_horizon past + 1 target per step, with
        # pred_horizon autoregressive steps → need obs_horizon + pred_horizon + 1
        # total frames so that the last step has a target.
        self.window_size = obs_horizon + pred_horizon + 1

        # Open zarr store
        root = zarr.open(zarr_path, mode="r")
        self._front = root["data"]["front"]           # zarr array, lazy
        self._tactile = root["data"]["left_tactile_camera_taxim"]  # zarr array, lazy
        self._action = root["data"]["action"]          # zarr array, lazy
        episode_ends = root["meta"]["episode_ends"][:]  # (num_episodes,) int64

        # Build list of (start, end) episode boundaries
        episodes: List[Tuple[int, int]] = []
        prev_end = 0
        for end in episode_ends:
            episodes.append((prev_end, int(end)))
            prev_end = int(end)

        # Collect all valid window start indices
        self._indices: List[int] = []
        for ep_start, ep_end in episodes:
            ep_len = ep_end - ep_start
            if ep_len < self.window_size:
                continue  # episode too short, skip entirely
            for start in range(ep_start, ep_end - self.window_size + 1):
                self._indices.append(start)

        # Action normalisation statistics
        if act_min is None or act_max is None:
            self.act_min, self.act_max = _compute_action_stats(root)
        else:
            self.act_min = act_min
            self.act_max = act_max

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self._indices[idx]
        end = start + self.window_size

        # Load raw arrays from zarr (returns numpy)
        front_raw = self._front[start:end]     # (T, 256, 256, 3) float32 [0,1]
        tactile_raw = self._tactile[start:end]  # (T, 320, 240, 3) float32 [0,1]
        action_raw = self._action[start:end]    # (T, 6 or 7) float32
        if action_raw.shape[1] < 7:
            action_raw = np.concatenate(
                [action_raw, np.zeros((len(action_raw), 7 - action_raw.shape[1]), dtype=np.float32)],
                axis=1,
            )

        T = self.window_size
        H = W = self.resize_scale

        # Resize images
        front_resized = np.stack(
            [_resize_image(front_raw[t], self.resize_scale) for t in range(T)],
            axis=0,
        )   # (T, H, W, 3)

        tactile_resized = np.stack(
            [_resize_image(tactile_raw[t], self.resize_scale) for t in range(T)],
            axis=0,
        )   # (T, H, W, 3)

        # Normalise images to [-1, 1]
        # The zarr arrays are float32 already in [0, 1] range
        front_norm = front_resized * 2.0 - 1.0    # (T, H, W, 3) in [-1, 1]
        tactile_norm = tactile_resized * 2.0 - 1.0  # (T, H, W, 3) in [-1, 1]

        # Normalise actions to [-1, 1]
        action_norm = _normalize_action(action_raw, self.act_min, self.act_max)  # (T, 7)

        # Convert to tensors in (T, C, H, W) format
        front_t = torch.from_numpy(
            front_norm.transpose(0, 3, 1, 2).copy()
        ).float()        # (T, 3, H, W)

        tactile_t = torch.from_numpy(
            tactile_norm.transpose(0, 3, 1, 2).copy()
        ).float()        # (T, 3, H, W)

        action_t = torch.from_numpy(action_norm.copy()).float()  # (T, 7)

        return {
            "front": front_t,
            "tactile": tactile_t,
            "action": action_t,
        }


# ---------------------------------------------------------------------------
# Multi-task combined dataset
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


class ManiFEELDataset(Dataset):
    """
    Combined dataset across all ManiFEEL tasks.

    Scans ``dataset_root`` for subdirectories matching known task names and
    creates a ManiFEELZarrTaskDataset for each.  All task datasets are
    concatenated into one flat index space.

    Args:
        dataset_root: path to the manifeel_data/data directory containing
                      task subdirectories (each is a zarr store)
        obs_horizon : number of past frames per context window
        pred_horizon: number of autoregressive prediction steps
        resize_scale: target image resolution
    """

    def __init__(
        self,
        dataset_root: str,
        obs_horizon: int,
        pred_horizon: int,
        resize_scale: int = 96,
    ) -> None:
        super().__init__()

        self._task_datasets: List[ManiFEELZarrTaskDataset] = []
        self._cum_lengths: List[int] = []

        cumulative = 0
        for task_dir in TASK_DIRS:
            zarr_path = os.path.join(dataset_root, task_dir)
            if not os.path.isdir(zarr_path):
                continue
            ds = ManiFEELZarrTaskDataset(
                zarr_path=zarr_path,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                resize_scale=resize_scale,
            )
            if len(ds) == 0:
                continue
            self._task_datasets.append(ds)
            cumulative += len(ds)
            self._cum_lengths.append(cumulative)

        if not self._task_datasets:
            raise RuntimeError(
                f"No valid task datasets found under '{dataset_root}'. "
                "Check that the path points to the manifeel_data/data directory."
            )

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Return (task_idx, local_idx) for a global index."""
        for task_idx, cum in enumerate(self._cum_lengths):
            if idx < cum:
                local_idx = idx if task_idx == 0 else idx - self._cum_lengths[task_idx - 1]
                return task_idx, local_idx
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, local_idx = self._locate(idx)
        return self._task_datasets[task_idx][local_idx]
