"""
train.py — Early Fusion multimodal world model training script
==============================================================

Usage (from repo root):
    python -m world_model_multimodal.early_fusion.train \
        [--config world_model_multimodal/early_fusion/configs/config.yml]

Or from within the early_fusion/ directory:
    python train.py [--config configs/config.yml]
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Local imports — all self-contained within early_fusion/
# ---------------------------------------------------------------------------
from dataset import build_combined_dataset
from diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from diffusion.inner_model import InnerModelConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(models_save_dir: str):
    """Return (checkpoint_dir, epoch) for the highest saved epoch, or (None, 0)."""
    if not os.path.isdir(models_save_dir):
        return None, 0
    pattern = re.compile(r"^checkpoint_epoch_(\d+)$")
    best_epoch, best_dir = 0, None
    for name in os.listdir(models_save_dir):
        m = pattern.match(name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_dir = epoch, os.path.join(models_save_dir, name)
    return best_dir, best_epoch


def save_checkpoint(
    epoch: int,
    nets: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    action_stats: dict,
) -> None:
    """Save model state dicts, optimizer state, epoch, and action normalization statistics."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name, model in nets.items():
        path = os.path.join(checkpoint_dir, f"{name}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
        print(f"  Saved {path}")
    # Save action stats as JSON (numpy arrays -> lists)
    stats_path = os.path.join(checkpoint_dir, "action_stats.json")
    stats_serializable = {
        k: {kk: vv.tolist() for kk, vv in v.items()}
        for k, v in {"action": action_stats}.items()
    }
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"  Saved {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    # ---- parse args ----
    parser = argparse.ArgumentParser(description="Early Fusion world model training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ---- hyper-parameters ----
    dataset_path: str = config["dataset_path"]
    models_save_dir: str = config["models_save_dir"]
    num_epochs: int = config["num_epochs"]
    obs_horizon: int = config["obs_horizon"]
    pred_horizon: int = config["pred_horizon"]
    batch_size: int = config["batch_size"]
    resize_scale: int = config["resize_scale"]
    lr: float = config["lr"]
    use_wandb: bool = config.get("wandb", False)
    phase_one_checkpoint: Optional[str] = config.get("phase_one_checkpoint", None)

    print(f"Device: {device}")
    print(f"Config: {config}")

    os.makedirs(models_save_dir, exist_ok=True)

    # ---- wandb (optional) ----
    if use_wandb:
        import wandb
        wandb.init(project="early_fusion_world_model", entity="projects-wpl", config=config, name="early_fusion")
    else:
        print("Warning: wandb is disabled")

    # ---- dataset ----
    print(f"\nLoading datasets from: {dataset_path}")
    combined_dataset, action_stats = build_combined_dataset(
        dataset_path=dataset_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        resize=resize_scale,
    )
    print(f"Total samples: {len(combined_dataset)}\n")

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # ---- model configuration ----
    # Early fusion: 6 channels per frame (3 RGB + 3 tactile)
    # Actions: 7-DoF robot
    # num_steps_conditioning: must equal obs_horizon so that the conditioning
    # window matches the act_emb expected input size.
    num_steps_conditioning = obs_horizon  # == 4 by default

    # Ensure cond_channels is divisible by num_steps_conditioning so that
    # the hidden_per_step = cond_channels // num_steps_conditioning is an integer.
    cond_channels = 256  # 256 / 4 = 64 per step

    inner_model_cfg = InnerModelConfig(
        img_channels=6,                        # 3 RGB + 3 tactile
        num_steps_conditioning=num_steps_conditioning,
        cond_channels=cond_channels,
        depths=[2, 2, 2, 2],
        channels=[96, 96, 96, 96],
        attn_depths=[False, False, True, True],
        num_actions=7,
        is_upsampler=False,
    )

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_model_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        noise_previous_obs=True,
        upsampling_factor=None,
    )

    sigma_dist_cfg = SigmaDistributionConfig(
        loc=-1.2,
        scale=1.2,
        sigma_min=2e-3,
        sigma_max=20.0,
    )

    diffusion_sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=3)

    # ---- instantiate model ----
    nets = nn.ModuleDict({"denoiser": Denoiser(denoiser_cfg)})
    nets = nets.to(device)
    nets["denoiser"].setup_training(sigma_dist_cfg)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(nets.parameters(), lr=lr)

    # ---- checkpoint loading ----
    start_epoch = 0
    if phase_one_checkpoint is not None:
        # Phase 2: warm-start from phase-1 weights; optimizer starts fresh
        state = torch.load(phase_one_checkpoint, map_location=device)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        nets["denoiser"].load_state_dict(state)
        print(f"Loaded phase-1 checkpoint: {phase_one_checkpoint}")
    else:
        # Auto-resume: find the latest checkpoint in models_save_dir
        latest_dir, latest_epoch = find_latest_checkpoint(models_save_dir)
        if latest_dir is not None:
            ckpt_path = os.path.join(latest_dir, "denoiser.pth")
            raw = torch.load(ckpt_path, map_location=device)
            if isinstance(raw, dict) and "model_state_dict" in raw:
                nets["denoiser"].load_state_dict(raw["model_state_dict"])
                optimizer.load_state_dict(raw["optimizer_state_dict"])
                start_epoch = raw.get("epoch", latest_epoch)
            else:
                nets["denoiser"].load_state_dict(raw)
                start_epoch = latest_epoch
            print(f"Resuming from epoch {start_epoch} ({latest_dir})")

    # ---- training loop ----
    print(f"Starting training: epochs {start_epoch + 1}–{num_epochs}, batch_size={batch_size}, lr={lr}")
    with tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epoch") as tglobal:
        for epoch_idx in tglobal:
            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch_idx})

            epoch_losses: List[float] = []
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for batch in tepoch:
                    # batch keys: 'front' (B,T,3,H,W), 'tactile' (B,T,3,H,W), 'action' (B,T,7)
                    loss, metrics = nets["denoiser"](batch, device)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    if use_wandb:
                        import wandb
                        wandb.log({"loss": loss_val, "epoch": epoch_idx})
                    epoch_losses.append(loss_val)
                    tepoch.set_postfix(loss=f"{loss_val:.4f}")

            mean_loss = float(np.mean(epoch_losses))
            tglobal.set_postfix(loss=f"{mean_loss:.4f}")

            # Save checkpoint every epoch
            checkpoint_dir = os.path.join(models_save_dir, f"checkpoint_epoch_{epoch_idx}")
            save_checkpoint(epoch_idx, nets, optimizer, checkpoint_dir, action_stats)

    print("Training complete.")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
