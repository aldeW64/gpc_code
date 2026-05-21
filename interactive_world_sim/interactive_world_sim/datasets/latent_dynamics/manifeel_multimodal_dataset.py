"""ManiFEEL multimodal zarr dataset for IWS LatentWorldModel multimodal fusion training.

Loads BOTH the front RGB camera AND the tactile camera from each task zarr store
and returns them as separate ``"rgb"`` and ``"tactile"`` observation keys so that
each fusion algorithm can merge them in whichever way it requires.

Zarr layout expected (each task subdirectory under dataset_path):
  data/front:                     (N, 256, 256, 3)  float32  HWC  [0, 1]
  data/left_tactile_camera_taxim: (N, 320, 240, 3)  float32  HWC  [0, 1]
  data/action:                    (N, 7)             float32
  meta/episode_ends:              (E,)               int64    exclusive end indices

Each sample emits:
  obs: {
    "rgb":     (T, 3, H, W)  float32  [0, 1]   (front camera, resized)
    "tactile": (T, 3, H, W)  float32  [0, 1]   (tactile camera, resized)
  }
  action: (T, 7)  float32  (unnormalised)

The corresponding LatentWorldModel variants (early / middle / late fusion) each
consume this batch and merge the two modalities in their own way.

Training commands — run from interactive_world_sim/ directory:

  # Early Fusion — Stage 1 (encoder + decoder, 6-channel input)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_early_fusion \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_early_stage1

  # Early Fusion — Stage 2 (dynamics)
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_early_fusion \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_early_stage2

  # Middle Fusion — Stage 1
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_middle_fusion \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_middle_stage1

  # Middle Fusion — Stage 2
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_middle_fusion \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_middle_stage2

  # Late Fusion — Stage 1
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_late_fusion \\
    algorithm.training_stage=1 \\
    algorithm.action_dim=7 \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_late_stage1

  # Late Fusion — Stage 2
  python main.py \\
    experiment=exp_latent_dyn \\
    dataset=manifeel_multimodal_dataset \\
    algorithm=latent_world_model_late_fusion \\
    algorithm.training_stage=2 \\
    algorithm.action_dim=7 \\
    "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \\
    wandb.entity=dummy \\
    wandb.mode=disabled \\
    +name=manifeel_late_stage2
"""

import copy
import os
from typing import Dict, List

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

# Keys used inside the unified ReplayBuffer
_RGB_KEY = "rgb"
_TACTILE_KEY = "tactile"
_ACTION_KEY = "action"

# ManiFEEL zarr array keys
_ZARR_RGB_KEY = "front"
_ZARR_TACTILE_KEY = "left_tactile_camera_taxim"


def _resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize (N, H, W, 3) float32 HWC images to (N, target_size, target_size, 3)."""
    h, w = images.shape[1], images.shape[2]
    if h == target_size and w == target_size:
        return images
    t = torch.from_numpy(images).permute(0, 3, 1, 2)
    t = F.interpolate(
        t,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return t.permute(0, 2, 3, 1).numpy()


def _build_multimodal_replay_buffer(
    dataset_path: str,
    resolution: int,
    action_dim: int = 7,
) -> ReplayBuffer:
    """Scan *dataset_path* for zarr task directories and concatenate both modalities.

    Tasks whose action array has a different number of columns than *action_dim*
    are silently skipped (ManiFEEL mixes 6-DOF and 7-DOF tasks).

    Returns a ReplayBuffer with keys ``"rgb"``, ``"tactile"``, and ``"action"``.
    """
    replay_buffer = ReplayBuffer.create_empty_numpy()

    task_dirs: List[str] = sorted(
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
    )

    if len(task_dirs) == 0:
        raise ValueError(
            f"No task subdirectories found in dataset_path='{dataset_path}'. "
            "Expected one directory per ManiFEEL task."
        )

    for task_name in task_dirs:
        task_path = os.path.join(dataset_path, task_name)
        if not os.path.isdir(os.path.join(task_path, "data")):
            continue
        if not os.path.isdir(os.path.join(task_path, "meta")):
            continue

        store = zarr.open(task_path, mode="r")

        for zarr_key, label in [(_ZARR_RGB_KEY, "RGB"), (_ZARR_TACTILE_KEY, "tactile")]:
            if zarr_key not in store["data"]:
                raise KeyError(
                    f"Task '{task_name}': expected key '{zarr_key}' in data group "
                    f"({label}), but found: {list(store['data'].keys())}"
                )

        rgb_images: np.ndarray = store["data"][_ZARR_RGB_KEY][:]
        tac_images: np.ndarray = store["data"][_ZARR_TACTILE_KEY][:]
        actions: np.ndarray = store["data"]["action"][:]
        episode_ends: np.ndarray = store["meta"]["episode_ends"][:]

        if actions.shape[1] != action_dim:
            print(
                f"[ManiFEELMultimodalDataset] skipping task '{task_name}': "
                f"action_dim={actions.shape[1]} != expected {action_dim}"
            )
            continue

        rgb_images = _resize_images(rgb_images.astype(np.float32), resolution)
        tac_images = _resize_images(tac_images.astype(np.float32), resolution)

        prev_end = 0
        for ep_end in episode_ends:
            replay_buffer.add_episode(
                {
                    _RGB_KEY: rgb_images[prev_end:ep_end],
                    _TACTILE_KEY: tac_images[prev_end:ep_end],
                    _ACTION_KEY: actions[prev_end:ep_end].astype(np.float32),
                }
            )
            prev_end = ep_end

    if replay_buffer.n_episodes == 0:
        raise ValueError(
            f"No valid episodes loaded from '{dataset_path}'. "
            "Check that task directories contain the expected zarr layout."
        )

    return replay_buffer


class ManiFEELMultimodalDataset(BaseImageDataset):
    """Multi-task ManiFEEL zarr dataset that returns both RGB and tactile observations.

    Configuration keys (DictConfig):
      dataset_path  : path to ManiFEEL task directory (relative paths resolved via Hydra)
      resolution    : target spatial resolution (default 96)
      horizon       : sequence length per training sample
      action_dim    : must be 7 for ManiFEEL
      obs_keys      : list of two strings — ["rgb", "tactile"] (used for normalizer keys)
      val_ratio     : fraction held out for validation (default 0.1)
      seed          : random seed for split (default 42)
      pad_before    : padding at episode start (default 1)
      pad_after     : padding at episode end (default 7)
      val_horizon   : sequence length for validation (defaults to horizon)
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

        # obs_keys should be ["rgb", "tactile"] for multimodal
        self.obs_keys: List[str] = list(cfg.obs_keys)

        dataset_path = str(cfg.dataset_path)
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(get_original_cwd(), dataset_path)
        self.dataset_path = dataset_path

        self.replay_buffer = _build_multimodal_replay_buffer(
            dataset_path=self.dataset_path,
            resolution=self.resolution,
            action_dim=int(cfg.get("action_dim", 7)),
        )

        n_eps = self.replay_buffer.n_episodes
        val_mask = get_val_mask(n_eps, val_ratio=self.val_ratio, seed=self.seed)
        train_mask = ~val_mask
        self.val_mask = val_mask
        self.train_mask = train_mask

        sampler_keys = [_RGB_KEY, _TACTILE_KEY, _ACTION_KEY]

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=train_mask,
            keys=sampler_keys,
            goal_sample="final",
        )
        self.val_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            keys=sampler_keys,
            goal_sample="final",
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Return a LinearNormalizer covering both modalities and actions.

        Both image modalities are in [0, 1] and are mapped to [-1, 1].
        Actions are 7-DOF continuous; mapped to [-1, 1] from data statistics.
        """
        normalizer = LinearNormalizer()
        normalizer[_RGB_KEY] = get_image_range_normalizer()
        normalizer[_TACTILE_KEY] = get_image_range_normalizer()
        action_data: np.ndarray = self.replay_buffer[_ACTION_KEY][:]
        stat = array_to_stats(action_data)
        normalizer[_ACTION_KEY] = get_range_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return len(self.val_sampler)
        return len(self.sampler)

    def _sample_to_data(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert raw sampler output to model-ready dict.

        Images: (T, H, W, 3) float32 [0, 1] -> (T, 3, H, W) float32 [0, 1]
        Actions: (T, 7) float32, unnormalised.
        """
        rgb = sample[_RGB_KEY].astype(np.float32)
        rgb = np.moveaxis(rgb, -1, 1)  # (T, H, W, 3) -> (T, 3, H, W)

        tac = sample[_TACTILE_KEY].astype(np.float32)
        tac = np.moveaxis(tac, -1, 1)

        actions = sample[_ACTION_KEY].astype(np.float32)

        return {
            "obs": {
                _RGB_KEY: torch.from_numpy(rgb),
                _TACTILE_KEY: torch.from_numpy(tac),
            },
            "action": torch.from_numpy(actions),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampler = self.val_sampler if self.is_val else self.sampler
        sample = sampler.sample_sequence(idx)
        return self._sample_to_data(sample)

    def get_validation_dataset(self) -> "ManiFEELMultimodalDataset":
        """Return a copy configured for validation."""
        val_set = copy.copy(self)
        val_set.is_val = True
        return val_set
