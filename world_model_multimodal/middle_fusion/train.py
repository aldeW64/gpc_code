"""
train.py — Training script for the Middle-Fusion multimodal world model.

Usage::

    # from repo root or from this directory
    python train.py [--config configs/config.yml]

The script:
  1. Loads config from YAML
  2. Builds ManiFEELDataset over all task zarr stores
  3. Instantiates MiddleFusionInnerModel + Denoiser
  4. Trains with AdamW at the specified lr
  5. Saves a checkpoint every epoch to <models_save_dir>/checkpoint_epoch_<N>/

Checkpoint format::

    {
        "epoch": int,
        "model_state_dict": OrderedDict,
        "optimizer_state_dict": OrderedDict,
        "config": dict,
    }
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from dataset import ManiFEELDataset
from diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from diffusion.inner_model import MiddleFusionInnerModel, MiddleFusionInnerModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    epoch: int,
    denoiser: Denoiser,
    optimizer: torch.optim.Optimizer,
    config: dict,
    checkpoint_dir: str,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "denoiser.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    denoiser: Denoiser,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Parse arguments ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train the Middle-Fusion multimodal world model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ---- Hyperparameters from config --------------------------------------
    dataset_path = config["dataset_path"]
    models_save_dir = config["models_save_dir"]
    num_epochs = config["num_epochs"]
    obs_horizon = config["obs_horizon"]
    pred_horizon = config["pred_horizon"]
    batch_size = config["batch_size"]
    resize_scale = config["resize_scale"]
    lr = float(config.get("lr", 1e-4))
    use_wandb = config.get("wandb", False)

    # Model architecture
    cond_channels = config.get("cond_channels", 256)
    depths = config.get("depths", [2, 2, 2, 2])
    channels = config.get("channels", [96, 96, 96, 96])
    attn_depths = config.get("attn_depths", [0, 0, 1, 1])

    # ---- Device ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- W&B ---------------------------------------------------------------
    if use_wandb:
        import wandb
        wandb.init(
            project="middle_fusion_world_model",
            config=config,
            name="middle_fusion_world_model",
        )
    else:
        print("W&B logging disabled.")

    # ---- Dataset -----------------------------------------------------------
    print(f"Loading dataset from: {dataset_path}")
    dataset = ManiFEELDataset(
        dataset_root=dataset_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        resize_scale=resize_scale,
    )
    print(f"Total samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # ---- Model -------------------------------------------------------------
    inner_model_cfg = MiddleFusionInnerModelConfig(
        img_channels=3,
        num_steps_conditioning=obs_horizon,
        num_actions=7,
        cond_channels=cond_channels,
        depths=depths,
        channels=channels,
        attn_depths=attn_depths,
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_model_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        noise_previous_obs=True,
    )

    sigma_dist_cfg = SigmaDistributionConfig(
        loc=-1.2,
        scale=1.2,
        sigma_min=2e-3,
        sigma_max=20.0,
    )

    denoiser = Denoiser(denoiser_cfg).to(device)
    denoiser.setup_training(sigma_dist_cfg)

    param_count = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr)

    # ---- Phase 2: load Phase-1 weights (fresh optimizer, new pred_horizon) ----
    start_epoch = 0
    phase_one_ckpt = config.get("phase_one_checkpoint", None)
    if phase_one_ckpt is not None:
        raw = torch.load(phase_one_ckpt, map_location=device)
        state = raw.get("model_state_dict", raw)
        denoiser.load_state_dict(state)
        print(f"Loaded phase-1 checkpoint: {phase_one_ckpt}")
    elif args.resume is not None:
        start_epoch = load_checkpoint(args.resume, denoiser, optimizer, device)
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training loop -----------------------------------------------------
    os.makedirs(models_save_dir, exist_ok=True)

    with tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epoch") as epoch_bar:
        for epoch_idx in epoch_bar:
            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch_idx})

            epoch_losses: list = []

            with tqdm(dataloader, desc="Batch", leave=False) as batch_bar:
                for batch in batch_bar:
                    loss, metrics = denoiser(batch, device)

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
                    optimizer.step()

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    batch_bar.set_postfix(loss=f"{loss_val:.4f}")

                    if use_wandb:
                        import wandb
                        wandb.log({"loss": loss_val, "epoch": epoch_idx})

            mean_loss = float(np.mean(epoch_losses))
            epoch_bar.set_postfix(loss=f"{mean_loss:.4f}")

            # ---- Save checkpoint ----------------------------------------
            checkpoint_dir = os.path.join(
                models_save_dir, f"checkpoint_epoch_{epoch_idx}"
            )
            save_checkpoint(epoch_idx, denoiser, optimizer, config, checkpoint_dir)

    print("Training complete.")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
