"""
ManiFEEL dataset loader for the GPC world model training pipelines.

ManiFEEL zarr layout (directory stores, no .zarr extension):
    <task_dir>/
        data/
            front              # (N, 256, 256, 3) float32 HWC  values in [0, 1]
            action             # (N, 7)           float32
            state              # (N, 7)           float32
        meta/
            episode_ends       # (E,) int64  exclusive end index of each episode

Returns batch dicts with keys:
    'image'  : (T, 3, resize_scale, resize_scale) float32 in [-1, 1]
    'action' : (T, 7)                             float32 in [-1, 1]

The key name 'image' (not 'front') is intentional so that the existing
denoiser.py forward pass works without any modification.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
import zarr


# ---------------------------------------------------------------------------
# Helpers shared with the existing PushT pipeline
# ---------------------------------------------------------------------------

def _get_data_stats(data: np.ndarray):
    """Compute per-dimension min/max statistics."""
    flat = data.reshape(-1, data.shape[-1])
    return {
        'min': np.min(flat, axis=0).astype(np.float32),
        'max': np.max(flat, axis=0).astype(np.float32),
    }


def _normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize to [-1, 1] using pre-computed min/max stats."""
    # Guard against zero-range dimensions
    denom = stats['max'] - stats['min']
    denom = np.where(denom == 0, 1.0, denom)
    ndata = (data - stats['min']) / denom   # [0, 1]
    ndata = ndata * 2.0 - 1.0              # [-1, 1]
    return ndata.astype(np.float32)


def _create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0) -> np.ndarray:
    """
    Build sliding-window indices that stay within episode boundaries.

    Returns array of shape (M, 4):
        [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    """
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx,
                            sample_start_idx, sample_end_idx])

    return np.array(indices, dtype=np.int64)


def _sample_sequence(train_data: dict, sequence_length: int,
                     buffer_start_idx: int, buffer_end_idx: int,
                     sample_start_idx: int, sample_end_idx: int) -> dict:
    """Extract a fixed-length window, padding at boundaries by repeating edge frames."""
    result = {}
    for key, arr in train_data.items():
        sample = arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            data = np.zeros(
                (sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        else:
            data = sample
        result[key] = data
    return result


def _resize_images_to_tensor(images_hwc: np.ndarray, resize_scale: int) -> np.ndarray:
    """
    Resize a batch of HWC uint8/float32 images to (T, 3, resize_scale, resize_scale).

    Input:  (T, H, W, 3) float32 in [0, 1]
    Output: (T, 3, resize_scale, resize_scale) float32 in [-1, 1]
    """
    # HWC -> CHW, then add batch dim: (T, H, W, 3) -> (T, 3, H, W)
    t = torch.from_numpy(images_hwc).permute(0, 3, 1, 2)  # (T, 3, H, W)
    # Bilinear resize
    t = F.interpolate(t, size=(resize_scale, resize_scale),
                      mode='bilinear', align_corners=False)
    # Normalize to [-1, 1]
    t = t * 2.0 - 1.0
    return t.numpy()


# ---------------------------------------------------------------------------
# Per-task dataset
# ---------------------------------------------------------------------------

class ManiFEELDataset(Dataset):
    """
    Single-task ManiFEEL dataset.

    Parameters
    ----------
    dataset_root : str
        Path to the ``dataset/manifeel/data`` directory.
    task_name : str
        Subdirectory name inside ``dataset_root`` (e.g. ``nutbolt_quan_July1``).
    obs_horizon : int
        Number of past context frames fed to the model.
    pred_horizon : int
        Total sliding-window length (= obs_horizon + future steps).
    resize_scale : int
        Square spatial resolution to resize images to (default 96).
    num_demos : int, optional
        Maximum number of episodes to load. Uses all if None or larger than
        the total number of episodes.
    action_stats : dict, optional
        Pre-computed action stats ``{'min': ndarray, 'max': ndarray}``.
        When None, stats are computed from this task's data.
    """

    def __init__(
            self,
            dataset_root: str,
            task_name: str,
            obs_horizon: int,
            pred_horizon: int,
            resize_scale: int = 96,
            num_demos: Optional[int] = None,
            action_stats: Optional[dict] = None,
    ) -> None:
        super().__init__()

        task_path = os.path.join(dataset_root, task_name)
        store = zarr.open(task_path, mode='r')

        # ---- episode boundaries ----------------------------------------
        episode_ends = store['meta']['episode_ends'][:]   # (E,) int64

        num_max_demos = episode_ends.shape[0]
        if num_demos is not None and num_demos < num_max_demos:
            num_max_demos = num_demos
        num_max_frames = int(episode_ends[num_max_demos - 1])
        episode_ends = episode_ends[:num_max_demos]

        # ---- load data into RAM (zarr is not safe across DataLoader workers) ---
        # front camera: (N, 256, 256, 3) float32 in [0, 1]
        images_raw = store['data']['front'][:num_max_frames]   # numpy HWC float32

        # 7-DOF actions: (N, 7) float32
        actions_raw = store['data']['action'][:num_max_frames]

        # ---- resize all images upfront to save per-sample work -------------
        # Output: (N, 3, resize_scale, resize_scale) float32 in [-1, 1]
        images_resized = _resize_images_to_tensor(images_raw, resize_scale)

        # ---- action normalization -----------------------------------------
        if action_stats is None:
            action_stats = _get_data_stats(actions_raw)
        actions_norm = _normalize_data(actions_raw, action_stats)

        # ---- sliding-window indices ---------------------------------------
        # sequence_length = pred_horizon (the denoiser iterates over the window)
        # pad_before = obs_horizon - 1 so windows can start at the first frame
        indices = _create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=0,
        )

        self.indices = indices
        self.action_stats = action_stats
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.resize_scale = resize_scale
        self.task_name = task_name

        # Store processed arrays
        self._data = {
            'image': images_resized,   # (N, 3, H, W)
            'action': actions_norm,    # (N, 7)
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = \
            self.indices[idx]

        # Extract window for images (already CHW)
        img_seq = self._data['image'][buffer_start_idx:buffer_end_idx]
        act_seq = self._data['action'][buffer_start_idx:buffer_end_idx]

        T = self.pred_horizon
        # Pad if near episode boundary
        def pad_seq(seq, shape_suffix, dtype):
            if sample_start_idx > 0 or sample_end_idx < T:
                out = np.zeros((T,) + shape_suffix, dtype=dtype)
                if sample_start_idx > 0:
                    out[:sample_start_idx] = seq[0]
                if sample_end_idx < T:
                    out[sample_end_idx:] = seq[-1]
                out[sample_start_idx:sample_end_idx] = seq
                return out
            return seq

        img_seq = pad_seq(img_seq, self._data['image'].shape[1:],
                          self._data['image'].dtype)
        act_seq = pad_seq(act_seq, self._data['action'].shape[1:],
                          self._data['action'].dtype)

        return {
            'image': torch.from_numpy(img_seq),   # (T, 3, H, W)
            'action': torch.from_numpy(act_seq),  # (T, 7)
        }


# ---------------------------------------------------------------------------
# Multi-task factory
# ---------------------------------------------------------------------------

def build_manifeel_dataset(
        dataset_root: str,
        obs_horizon: int,
        pred_horizon: int,
        resize_scale: int = 96,
        num_demos: Optional[int] = None,
) -> ConcatDataset:
    """
    Scan ``dataset_root`` for ManiFEEL task directories (no .zarr filter —
    ManiFEEL dirs have no extension) and return a ``ConcatDataset`` over all
    tasks.  Each task computes its own action normalization statistics.

    Parameters
    ----------
    dataset_root : str
        Path to ``dataset/manifeel/data``.
    obs_horizon : int
        Number of conditioning frames.
    pred_horizon : int
        Sliding-window length.
    resize_scale : int
        Target image resolution (default 96).
    num_demos : int, optional
        Maximum episodes per task.  None means use all.

    Returns
    -------
    ConcatDataset
        Concatenation of all per-task ``ManiFEELDataset`` objects.
    """
    datasets: List[ManiFEELDataset] = []

    for entry in sorted(os.listdir(dataset_root)):
        full_path = os.path.join(dataset_root, entry)
        # Skip zip archives and any non-directory entries
        if not os.path.isdir(full_path):
            continue
        # Skip hidden directories
        if entry.startswith('.'):
            continue
        # Verify it looks like a ManiFEEL zarr store (has data/ and meta/)
        if not (os.path.isdir(os.path.join(full_path, 'data')) and
                os.path.isdir(os.path.join(full_path, 'meta'))):
            continue

        print(f"Loading ManiFEEL task: {entry}")
        ds = ManiFEELDataset(
            dataset_root=dataset_root,
            task_name=entry,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            resize_scale=resize_scale,
            num_demos=num_demos,
        )
        print(f"  -> {len(ds)} samples")
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(
            f"No valid ManiFEEL task directories found in {dataset_root}. "
            "Expected directories with data/ and meta/ subdirectories."
        )

    return ConcatDataset(datasets)
