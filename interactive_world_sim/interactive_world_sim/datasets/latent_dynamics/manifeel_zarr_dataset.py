"""ManiFEEL zarr dataset for interactive_world_sim LatentWorldModel training.

Zarr layout expected (each task subdirectory under dataset_path):
  data/front:            (N, 256, 256, 3)  float32  HWC  values in [0, 1]
  data/action:           (N, 7)            float32  7-DOF actions
  meta/episode_ends:     (E,)              int64    exclusive end indices

Multiple task subdirectories are concatenated into a single ReplayBuffer.
Episodes from all tasks are combined, and a single train/val split is applied.

Each sample emits:
  obs:    {cfg.obs_keys[0]: (T, 3, H, W)  float32  [0, 1]}  (resized to cfg.resolution)
  action: (T, 7)  float32  (unnormalised — the normaliser in LatentWorldModel
                             maps these to [-1, 1])

Training commands (run from interactive_world_sim/ directory):

  # Stage 1: encoder + decoder
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_dataset \\
    algorithm=latent_world_model \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_stage1

  # Stage 2: dynamics (replace <date>/<time>/<step> with actual values)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_dataset \\
    algorithm=latent_world_model \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_stage2
"""

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler, get_val_mask

from .base_dataset import BaseImageDataset


def _resize_images(
    images: np.ndarray, target_size: int
) -> np.ndarray:
    """Resize (N, H, W, 3) float32 HWC images to (N, target_size, target_size, 3).

    Uses bilinear interpolation via torch.nn.functional.interpolate.
    If the images are already at the target resolution, returns the input array
    unchanged (no copy).
    """
    h, w = images.shape[1], images.shape[2]
    if h == target_size and w == target_size:
        return images

    # (N, H, W, 3) -> (N, 3, H, W)
    t = torch.from_numpy(images).permute(0, 3, 1, 2)
    t = F.interpolate(
        t,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    # (N, 3, H, W) -> (N, H, W, 3)
    return t.permute(0, 2, 3, 1).numpy()


def _build_replay_buffer(
    dataset_path: str,
    resolution: int,
    camera_key: str,
) -> ReplayBuffer:
    """Scan *dataset_path* for zarr task directories and concatenate them.

    Each immediate subdirectory that is a valid zarr group (i.e. contains
    both ``data/`` and ``meta/episode_ends``) is treated as one task.
    Arrays are loaded into memory, resized, and added episode-by-episode to
    a single numpy-backed ReplayBuffer.

    Parameters
    ----------
    dataset_path:
        Absolute path to the directory containing task zarr stores.
    resolution:
        Target spatial resolution; images are resized to (resolution, resolution).
    camera_key:
        Name of the image array inside ``data/`` (e.g. ``"front"``).

    Returns
    -------
    ReplayBuffer
        A numpy-backed ReplayBuffer with keys ``"image"`` and ``"action"``.
    """
    replay_buffer = ReplayBuffer.create_empty_numpy()

    task_dirs: List[str] = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and not d.startswith(".")
    )

    if len(task_dirs) == 0:
        raise ValueError(
            f"No task subdirectories found in dataset_path='{dataset_path}'. "
            "Expected one directory per ManiFEEL task."
        )

    for task_name in task_dirs:
        task_path = os.path.join(dataset_path, task_name)
        # Validate it looks like a zarr store with the expected layout.
        if not os.path.isdir(os.path.join(task_path, "data")):
            continue
        if not os.path.isdir(os.path.join(task_path, "meta")):
            continue

        store = zarr.open(task_path, mode="r")

        if camera_key not in store["data"]:
            raise KeyError(
                f"Task '{task_name}': expected key '{camera_key}' in data group, "
                f"but found: {list(store['data'].keys())}"
            )

        # Load images: (N, H, W, 3) float32 [0, 1]
        images: np.ndarray = store["data"][camera_key][:]  # (N, 256, 256, 3)
        actions: np.ndarray = store["data"]["action"][:]    # (N, 7)
        episode_ends: np.ndarray = store["meta"]["episode_ends"][:]  # (E,)

        # Resize images if needed.
        images = _resize_images(images.astype(np.float32), resolution)

        # Determine per-episode boundaries (exclusive end indices are absolute).
        prev_end = 0
        for ep_end in episode_ends:
            ep_images = images[prev_end:ep_end]   # (L, H, W, 3)
            ep_actions = actions[prev_end:ep_end]  # (L, 7)
            replay_buffer.add_episode(
                {
                    "image": ep_images,
                    "action": ep_actions.astype(np.float32),
                }
            )
            prev_end = ep_end

    if replay_buffer.n_episodes == 0:
        raise ValueError(
            f"No valid episodes loaded from '{dataset_path}'. "
            "Check that task directories contain the expected zarr layout."
        )

    return replay_buffer


class ManiFEELZarrDataset(BaseImageDataset):
    """Multi-task ManiFEEL zarr dataset for LatentWorldModel training.

    Accepts a Hydra DictConfig with the following keys:

    dataset_path : str
        Path to the directory containing ManiFEEL task zarr stores (each
        subdirectory is one task).  Relative paths are resolved with respect
        to the original working directory (``hydra.utils.get_original_cwd()``).
    resolution : int
        Target image resolution (images are resized to resolution×resolution).
    horizon : int
        Number of timesteps per training sample.
    action_dim : int
        Action dimensionality (should be 7 for ManiFEEL).
    camera_key : str, optional
        Name of the camera array inside the zarr store's ``data`` group.
        Defaults to ``front``.
    obs_keys : list of str
        Batch observation key consumed by LatentWorldModel.  Defaults to
        ``['image']`` in the provided config.
    val_ratio : float, optional
        Fraction of episodes held out for validation (default 0.1).
    seed : int, optional
        Random seed for the train/val split (default 42).
    pad_before : int, optional
        Padding at the start of each episode (default 1).
    pad_after : int, optional
        Padding at the end of each episode (default 7).
    val_horizon : int, optional
        Sequence length for validation samples (defaults to ``horizon``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.resolution = int(cfg.resolution)
        self.horizon = int(cfg.horizon)
        self.val_horizon = int(cfg.get("val_horizon", cfg.horizon))
        self.pad_before = int(cfg.get("pad_before", 1))
        self.pad_after = int(cfg.get("pad_after", 7))
        self.val_ratio = float(cfg.get("val_ratio", 0.1))
        self.seed = int(cfg.get("seed", 42))

        obs_keys = list(cfg.obs_keys)
        self.obs_key: str = obs_keys[0]
        self.camera_key: str = str(cfg.get("camera_key", "front"))

        # Resolve dataset_path (may be relative to the original cwd under Hydra).
        dataset_path = str(cfg.dataset_path)
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(get_original_cwd(), dataset_path)
        self.dataset_path = dataset_path

        # Build a unified ReplayBuffer from all task directories.
        self.replay_buffer = _build_replay_buffer(
            dataset_path=self.dataset_path,
            resolution=self.resolution,
            camera_key=self.camera_key,
        )

        # Train / val split at the episode level.
        n_eps = self.replay_buffer.n_episodes
        val_mask = get_val_mask(n_eps, val_ratio=self.val_ratio, seed=self.seed)
        train_mask = ~val_mask
        self.val_mask = val_mask
        self.train_mask = train_mask

        # Sequence samplers — keys must match what _build_replay_buffer stored.
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=train_mask,
            keys=["image", "action"],
            goal_sample="final",
        )
        self.val_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            keys=["image", "action"],
            goal_sample="final",
        )

    # ------------------------------------------------------------------
    # BaseImageDataset interface
    # ------------------------------------------------------------------

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Return a LinearNormalizer for images and actions.

        Images are already in [0, 1]; map to [-1, 1] via
        ``get_image_range_normalizer()``.

        Actions are 7-DOF continuous values; map to [-1, 1] using data
        statistics computed from the full dataset.
        """
        normalizer = LinearNormalizer()
        # Images: float [0, 1] → [-1, 1]
        normalizer[self.obs_key] = get_image_range_normalizer()
        # Actions: fit min/max from dataset
        action_data: np.ndarray = self.replay_buffer["action"][:]
        stat = array_to_stats(action_data)
        normalizer["action"] = get_range_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return len(self.val_sampler)
        return len(self.sampler)

    def _sample_to_data(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert a raw sampler output to a model-ready dict.

        Images are (T, H, W, 3) float32 [0, 1] → (T, 3, H, W) float32 [0, 1].
        Actions are (T, 7) float32, returned unnormalised (the LatentWorldModel
        applies its own normaliser at training time).
        """
        # (T, H, W, 3) float32 [0, 1] → (T, 3, H, W)
        imgs = sample["image"].astype(np.float32)
        imgs = np.moveaxis(imgs, -1, 1)  # HWC → CHW per frame

        actions = sample["action"].astype(np.float32)  # (T, 7)

        return {
            "obs": {self.obs_key: torch.from_numpy(imgs)},
            "action": torch.from_numpy(actions),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampler = self.val_sampler if self.is_val else self.sampler
        sample = sampler.sample_sequence(idx)
        return self._sample_to_data(sample)

    def get_validation_dataset(self) -> "ManiFEELZarrDataset":
        """Return a copy of this dataset configured for validation."""
        val_set = copy.copy(self)
        val_set.is_val = True
        return val_set
