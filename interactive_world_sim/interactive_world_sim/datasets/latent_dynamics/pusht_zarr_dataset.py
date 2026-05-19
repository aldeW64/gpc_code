"""PushT zarr dataset for interactive_world_sim LatentWorldModel training.

Zarr layout expected:
  data/img:            (N, H, W, 3)  uint8   HWC
  data/action:         (N, 2)        float32 pixel coords [0, 511]
  meta/episode_ends:   (E,)          int64

Each sample emits:
  obs:    {"image": (T, 3, H, W)  float32  [0, 1]}
  action: (T, 2)  float32  (unnormalised pixel coords — the normaliser in
                             LatentWorldModel maps these to [-1, 1])
"""

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
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


class PushTZarrDataset(BaseImageDataset):
    """PushT zarr dataset for LatentWorldModel training."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.resolution = int(cfg.resolution)
        self.horizon = int(cfg.horizon)
        self.val_horizon = int(cfg.get("val_horizon", cfg.horizon))
        self.pad_before = int(cfg.get("pad_before", 1))
        self.pad_after = int(cfg.get("pad_after", 7))
        self.val_ratio = float(cfg.get("val_ratio", 0.1))
        self.seed = int(cfg.get("seed", 42))

        zarr_path = str(cfg.dataset_path)
        if not os.path.isabs(zarr_path):
            zarr_path = os.path.join(get_original_cwd(), zarr_path)
        z = zarr.open(zarr_path, mode="r")
        self.replay_buffer = ReplayBuffer(z)

        n_eps = self.replay_buffer.n_episodes
        val_mask = get_val_mask(n_eps, val_ratio=self.val_ratio, seed=self.seed)
        train_mask = ~val_mask
        self.val_mask = val_mask
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=train_mask,
            keys=["img", "action"],
            goal_sample="final",
        )
        self.val_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            keys=["img", "action"],
            goal_sample="final",
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # Images are float [0,1] — map to [-1,1] for the model
        normalizer["image"] = get_image_range_normalizer()
        # Actions: pixel coords in roughly [0,511] — map to [-1,1]
        action_data = self.replay_buffer["action"][:]
        stat = array_to_stats(action_data)
        normalizer["action"] = get_range_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return len(self.val_sampler)
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # img: (T, H, W, 3) uint8  →  (T, 3, H, W) float32 [0,1]
        imgs = sample["img"].astype(np.float32) / 255.0
        imgs = np.moveaxis(imgs, -1, 1)

        actions = sample["action"].astype(np.float32)  # (T, 2)

        return {
            "obs": {"image": torch.from_numpy(imgs)},
            "action": torch.from_numpy(actions),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampler = self.val_sampler if self.is_val else self.sampler
        sample = sampler.sample_sequence(idx)
        return self._sample_to_data(sample)

    def get_validation_dataset(self) -> "PushTZarrDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        return val_set
