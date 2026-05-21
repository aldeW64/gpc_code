"""
Phase-1 world model training on ManiFEEL data.

Differences from train.py (PushT):
  - Uses ManiFEELDataset / build_manifeel_dataset from dataset_manifeel
    (no .zarr filter; reads 'data/front' + 'data/action' from each task dir)
  - Imports Denoiser from diffusion.denoiser_manifeel, which in turn uses
    inner_model_manifeel (configurable action_dim, default 7)
  - Config key is 'dataset_path' (points to dataset/manifeel/data)
  - action_dim is set to 7 in InnerModelConfig
  - Everything else — sigma schedule, UNet shape, optimiser, save cadence —
    is identical to the PushT phase-1 training script.

Run from repo root:
    python -m world_model_train_phase_one.train_manifeel \
        --config world_model_train_phase_one/configs/train_manifeel_phase_one_config.yml
Or from within world_model_train_phase_one/:
    python train_manifeel.py --config configs/train_manifeel_phase_one_config.yml
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional
import argparse
import wandb
import os
import yaml

try:
    # Supports running from within world_model_train_phase_one/.
    from dataset_manifeel import build_manifeel_dataset
    from utils import save
    from diffusion.denoiser_manifeel import (
        Denoiser,
        DenoiserConfig,
        InnerModelConfig,
    )
except ModuleNotFoundError:
    # Supports running from repo root with python -m world_model_train_phase_one.train_manifeel.
    from .dataset_manifeel import build_manifeel_dataset
    from .utils import save
    from .diffusion.denoiser_manifeel import (
        Denoiser,
        DenoiserConfig,
        InnerModelConfig,
    )


def main():
    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser(
        description='Phase-1 world model training on ManiFEEL dataset.')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/train_manifeel_phase_one_config.yml',
        help='Path to the configuration YAML file.',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a denoiser.pth checkpoint to resume from.',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(script_dir, path))

    num_epochs = config['num_epochs']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    batch_size = config['batch_size']
    dataset_path = resolve_path(config['dataset_path'])   # path to dataset/manifeel/data
    models_save_dir = resolve_path(config['models_save_dir'])
    resize_scale = config.get('resize_scale', 96)

    if config.get('wandb', False):
        wandb.init(
            project='world_model_train_phase_one_manifeel',
            config=config,
            name='world_model_train_phase_one_manifeel',
        )
    else:
        print("warning: wandb flag set to False")

    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    # ---- Build dataset ---------------------------------------------------
    # build_manifeel_dataset scans all task subdirectories (no .zarr filter)
    combined_dataset = build_manifeel_dataset(
        dataset_root=dataset_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        resize_scale=resize_scale,
        num_demos=num_train_demos,
    )
    print(f"Total training samples: {len(combined_dataset)}")

    # ---- DataLoader ------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = nn.ModuleDict({})

    # ---- Model configuration ---------------------------------------------
    # num_steps_conditioning must equal obs_horizon so the act_emb linear
    # maps (b, obs_horizon, action_dim) -> (b, obs_horizon, cond_channels//obs_horizon)
    # then flatten -> (b, cond_channels).
    @dataclass
    class SigmaDistributionConfig:
        loc = -1.2
        scale = 1.2
        sigma_min = 2e-3
        sigma_max = 20

    sigma_distribution_config = SigmaDistributionConfig()

    inner_model_config = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=obs_horizon,   # must equal obs_horizon
        cond_channels=256,
        depths=[2, 2, 2, 2],
        channels=[96, 96, 96, 96],
        attn_depths=[0, 0, 1, 1],
        action_dim=7,                         # 7-DOF ManiFEEL actions
        num_actions=None,
        is_upsampler=None,
    )

    denoiser_config = DenoiserConfig(
        inner_model=inner_model_config,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        noise_previous_obs=True,
        upsampling_factor=None,
    )

    @dataclass
    class DiffusionSamplerConfig:
        num_steps_denoising: int
        sigma_min: float = 2e-3
        sigma_max: float = 5
        rho: int = 7
        order: int = 1
        s_churn: float = 0
        s_tmin: float = 0
        s_tmax: float = float('inf')
        s_noise: float = 1
        s_cond: float = 0

    diffusion_sampler_config = DiffusionSamplerConfig(num_steps_denoising=3)

    nets['denoiser'] = Denoiser(denoiser_config)
    nets = nets.to(device)
    nets['denoiser'].setup_training(sigma_distribution_config)

    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4)

    # ---- Checkpoint resume -----------------------------------------------
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        nets['denoiser'].load_state_dict(state)
        try:
            start_epoch = int(
                os.path.basename(os.path.dirname(args.resume)).split('_')[-1]
            )
        except (ValueError, IndexError):
            pass
        print(f"Resuming phase-1 from epoch {start_epoch} ({args.resume})")

    # ---- Training loop ---------------------------------------------------
    with tqdm(range(start_epoch + 1, num_epochs + 1), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            if config.get('wandb', False):
                wandb.log({'epoch': epoch_idx})
            epoch_loss = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    loss, _ = nets['denoiser'](nbatch, device)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_cpu = loss.item()
                    if config.get('wandb', False):
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # Save checkpoint
            if epoch_idx in [1, 5, 10, num_epochs] or (epoch_idx % 5 == 0):
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(
                    models_save_dir, epoch_idx)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(nets, checkpoint_dir)


if __name__ == '__main__':
    main()
