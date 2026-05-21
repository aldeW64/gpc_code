"""
Training script for the Late Fusion (Score Composition) multimodal world model.

Usage (run from repo root OR from this directory):
    python train.py [--config configs/config.yml]

The script:
  1. Loads all manifeel zarr task stores from dataset_path.
  2. Instantiates a LateFusionDenoiser with two InnerModel experts (RGB + tactile).
  3. Trains both experts jointly with a single AdamW optimizer.
  4. Saves checkpoints every epoch (or at the configured interval).
"""

import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from dataset import build_combined_dataset
from diffusion.denoiser import (
    LateFusionDenoiser,
    LateFusionDenoiserConfig,
    SigmaDistributionConfig,
)
from diffusion.diffusion_sampler import (
    LateFusionDiffusionSampler,
    LateFusionDiffusionSamplerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    name: str = "late_fusion_denoiser",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[train] Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Train the Late Fusion multimodal world model.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yml',
        help='Path to the YAML configuration file.',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ---- Config extraction ----
    dataset_path = config['dataset_path']
    models_save_dir = config['models_save_dir']
    num_epochs = config['num_epochs']
    obs_horizon = config['obs_horizon']
    pred_horizon = config['pred_horizon']
    batch_size = config['batch_size']
    resize_scale = config['resize_scale']
    lr = config.get('lr', 1e-4)
    use_wandb = config.get('wandb', False)
    loss_alpha = config.get('loss_alpha', 0.5)
    cond_channels = config.get('cond_channels', 256)
    depths = config.get('depths', [2, 2, 2, 2])
    channels = config.get('channels', [96, 96, 96, 96])
    attn_depths = config.get('attn_depths', [0, 0, 1, 1])
    save_every = config.get('save_every', 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ---- W&B ----
    if use_wandb:
        if not _WANDB_AVAILABLE:
            print("[train] WARNING: wandb not installed, skipping W&B logging.")
            use_wandb = False
        else:
            wandb.init(
                project="late_fusion_world_model",
                config=config,
                name="late_fusion_multimodal",
            )
    else:
        print("[train] W&B logging disabled.")

    # ---- Dataset ----
    print("[train] Building dataset ...")
    combined_dataset = build_combined_dataset(
        dataset_root=dataset_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        resize_scale=resize_scale,
    )
    print(f"[train] Total windows: {len(combined_dataset)}")

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # ---- Model ----
    denoiser_cfg = LateFusionDenoiserConfig(
        img_channels=3,
        num_steps_conditioning=obs_horizon,
        cond_channels=cond_channels,
        action_dim=7,
        depths=depths,
        channels=channels,
        attn_depths=attn_depths,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        noise_previous_obs=True,
        loss_alpha=loss_alpha,
    )

    denoiser = LateFusionDenoiser(denoiser_cfg).to(device)

    sigma_dist_cfg = SigmaDistributionConfig(
        loc=-1.2,
        scale=1.2,
        sigma_min=2e-3,
        sigma_max=20.0,
    )
    denoiser.setup_training(sigma_dist_cfg)

    num_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {num_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr)

    # ---- Phase 2: load Phase-1 weights (fresh optimizer, new pred_horizon) ----
    start_epoch = 0
    phase_one_ckpt = config.get('phase_one_checkpoint', None)
    if phase_one_ckpt is not None:
        raw = torch.load(phase_one_ckpt, map_location=device)
        state = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
        denoiser.load_state_dict(state)
        print(f'[train] Loaded phase-1 checkpoint: {phase_one_ckpt}')
    else:
        # Auto-resume: find the latest checkpoint in models_save_dir
        latest_dir, latest_epoch = find_latest_checkpoint(models_save_dir)
        if latest_dir is not None:
            ckpt_path = os.path.join(latest_dir, 'late_fusion_denoiser.pth')
            raw = torch.load(ckpt_path, map_location=device)
            if isinstance(raw, dict) and 'model_state_dict' in raw:
                denoiser.load_state_dict(raw['model_state_dict'])
                optimizer.load_state_dict(raw['optimizer_state_dict'])
                start_epoch = raw.get('epoch', latest_epoch)
            else:
                denoiser.load_state_dict(raw)
                start_epoch = latest_epoch
            print(f'[train] Resuming from epoch {start_epoch} ({latest_dir})')

    os.makedirs(models_save_dir, exist_ok=True)

    # ---- Training loop ----
    global_step = 0
    with tqdm(range(start_epoch + 1, num_epochs + 1), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            if use_wandb:
                wandb.log({'epoch': epoch_idx})

            epoch_loss: list = []
            epoch_loss_rgb: list = []
            epoch_loss_tac: list = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch in tepoch:
                    loss, metrics = denoiser(batch, device)

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
                    optimizer.step()

                    loss_val = metrics['loss']
                    epoch_loss.append(loss_val)
                    epoch_loss_rgb.append(metrics['loss_rgb'])
                    epoch_loss_tac.append(metrics['loss_tac'])

                    if use_wandb:
                        wandb.log({
                            'loss': loss_val,
                            'loss_rgb': metrics['loss_rgb'],
                            'loss_tac': metrics['loss_tac'],
                            'step': global_step,
                        })

                    tepoch.set_postfix(
                        loss=f"{loss_val:.4f}",
                        rgb=f"{metrics['loss_rgb']:.4f}",
                        tac=f"{metrics['loss_tac']:.4f}",
                    )
                    global_step += 1

            mean_loss = float(np.mean(epoch_loss))
            mean_rgb = float(np.mean(epoch_loss_rgb))
            mean_tac = float(np.mean(epoch_loss_tac))

            tglobal.set_postfix(
                loss=f"{mean_loss:.4f}",
                rgb=f"{mean_rgb:.4f}",
                tac=f"{mean_tac:.4f}",
            )

            if use_wandb:
                wandb.log({
                    'epoch_loss': mean_loss,
                    'epoch_loss_rgb': mean_rgb,
                    'epoch_loss_tac': mean_tac,
                    'epoch': epoch_idx,
                })

            # ---- Save checkpoint ----
            if epoch_idx % save_every == 0:
                checkpoint_dir = os.path.join(models_save_dir, f'checkpoint_epoch_{epoch_idx}')
                save_checkpoint(epoch_idx, denoiser, optimizer, checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(num_epochs, denoiser, optimizer, os.path.join(models_save_dir, 'checkpoint_final'))
    print("[train] Training complete.")


if __name__ == '__main__':
    main()
