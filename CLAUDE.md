# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the official implementation of "Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling" (IEEE RA-L 2025). The framework consists of two main components:

1. **Diffusion-based Action Policy**: Generates candidate action sequences using a generative diffusion model
2. **Predictive World Model**: Learns environment dynamics to evaluate and refine candidate trajectories

At inference time, the world model enhances policy performance through trajectory prediction and ranking/optimization.

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate gpc
```

The project uses Python 3.12 with PyTorch, diffusers, and robotics-related libraries (gym, pymunk, shapely).

## Architecture

### Directory Structure

- **diffusion_policy_training/**: Trains the diffusion-based action policy that generates action sequences
- **world_model_train_phase_one/**: Single-step world model warmup training (stabilizes early training)
- **world_model_train_phase_two/**: Multi-step world model training (enables long-horizon trajectory evaluation)
- **gpc_rank_evaluation/**: Evaluation pipeline using trajectory ranking (samples candidates, scores via world model, executes best)
- **gpc_opt_evaluation/**: Evaluation pipeline using trajectory optimization (iteratively refines action sequences via world model)
- **gpc_wam_evaluation/**: Evaluation pipeline that scores candidates by `-MSE(predicted_final_image, goal_image)`; supports three world model backends (`world_model_type: gpc | wam | iws`), two planners (diffusion / MPPI), and online env or offline zarr modes
- **wam/**: Symlink to the Ctrl-World repository (`Projects/Ctrl-World`). Contains the CrtlWorld video-diffusion world model, its SVD/CLIP base models, training scripts, and pre-trained checkpoints
- **interactive_world_sim/**: Separate git repo containing the `LatentWorldModel` — a 3-stage (CNN encoder → causal latent dynamics → CM-decoder) world model trained on PushT zarr. Used as an alternative scoring backend (`world_model_type: iws`) in both `gpc_wam_evaluation` and `gpc_rank_evaluation`
- **all_checkpoints/**: Pre-trained checkpoints for policy, world model, and reward predictors

### Shared Components

Each training and evaluation module contains:

- **models.py**: Neural network architectures (denoiser, UNet-based components)
- **utils.py**: Dataset loading, data normalization, checkpoint save/load utilities
- **pusht_env.py**: PushT simulation environment (tasks, rendering, reward computation)
- **diffusion/**: Diffusion sampling and denoiser implementations
- **configs/**: YAML configuration files for hyperparameters and checkpoint paths
- **data/**: Environment-specific metadata and domain definitions

### Configuration System

All modules are configuration-driven via YAML files in `configs/` directories:

- `pred_horizon`: Number of steps to predict/plan ahead
- `obs_horizon`: Number of past observations for conditioning
- `action_horizon`: Length of action sequences
- `batch_size`: Training batch size
- `num_epochs`/`num_diffusion_iters`: Training duration
- `resize_scale`: Image resolution
- Checkpoint paths for policy, world model, and reward predictors
- `wandb`: Whether to log to Weights & Biases

## Common Commands

### Training

```bash
# Train diffusion action policy
cd diffusion_policy_training
python train_model.py [--config configs/train_config.yml]

# Train world model - Phase 1 (single-step warmup)
cd world_model_train_phase_one
python train.py [--config configs/train_phase_one_config.yml]

# Train world model - Phase 2 (multi-step)
cd world_model_train_phase_two
python train.py [--config configs/train_phase_two_config.yml]
```

### Evaluation

All evaluation scripts must be run from the **repo root** using `-m` (so relative imports resolve correctly):

```bash
# GPC-RANK: Ranking-based trajectory selection
python -m gpc_rank_evaluation.gpc_rank_evaluation [--config gpc_rank_evaluation/configs/gpc_rank_evaluation_config.yml]

# GPC-OPT: Gradient-based trajectory optimization (still uses legacy direct-script style)
cd gpc_opt_evaluation
python gpc_opt_evaluation.py [--config configs/gpc_opt_evaluation_config.yml]

# GPC-WAM with GPC world model (default; uses all_checkpoints/world_model_checkpoint)
python -m gpc_wam_evaluation.gpc_wam_evaluation --num_episodes 100

# GPC-WAM with Ctrl-World (WAM) backend
python -m gpc_wam_evaluation.gpc_wam_evaluation --world_model_type wam [--wam_ckpt_path <path/to/checkpoint.pt>]

# GPC-WAM quick offline sanity check (zarr ground-truth actions, no env needed)
python -m gpc_wam_evaluation.gpc_wam_evaluation \
  --eval_mode offline_dataset \
  --offline_zarr_path dataset/world_model_data/dataset_domain/all_data/domain18.zarr \
  --num_episodes 20

# GPC-WAM online evaluation with MPPI planner
python -m gpc_wam_evaluation.gpc_wam_evaluation --planner_mode mppi --num_episodes 100
```

### WAM (World Action Model / Ctrl-World)

`wam/` is a symlink to the Ctrl-World video-diffusion model. It acts as an alternative world model for GPC-WAM evaluation. Key facts:

- **Architecture**: Action-conditioned SVD (Stable Video Diffusion). The action encoder (3-layer MLP) maps `(num_history+num_frames, action_dim=7)` → `(T, 1024)` embeddings that condition the UNet via cross-attention.
- **PushT adaptation**: PushT uses 2D pixel actions. The model keeps `action_dim=7`; the 2D actions are placed in dims `[0:2]` and dims `[2:7]` are zero-padded. This is handled identically in the dataset loader (`wam/dataset/dataset_pusht_zarr.py`) and the eval adapter (`gpc_wam_evaluation/wam_adapter.py`).
- **Single-view tiling**: PushT renders a single square view. Ctrl-World expects a 3-view height-stacked latent layout, so each latent is tiled 3× in the height dimension before being fed to the model.
- **Checkpoint format**: `torch.save(model.state_dict(), path)` → `.pt` file (~9 GB). Loaded with `strict=False` to allow partial weight loading when fine-tuning.
- **Checkpoint search order** (auto-resolved when no explicit path given):
  1. `./wam/ckpt/pusht_finetuned/` — PushT fine-tuned (preferred)
  2. `./wam/ckpt/` — DROID pre-trained fallback

#### Training WAM on PushT

Run from the **repo root**:

```bash
python wam/scripts/train_wm_pusht_zarr.py \
  --zarr_path dataset/world_model_data/dataset_domain/all_data/domain18.zarr \
  --svd_model_path wam/svd \
  --clip_model_path wam/clip \
  --ckpt_path wam/ckpt/checkpoint-10000.pt \
  --output_dir wam/ckpt/pusht_finetuned \
  --num_samples 20000 \
  --max_train_steps 10000 \
  --checkpointing_steps 2000 \
  --mixed_precision fp16 \
  --no_wandb
```

To submit via SLURM (Kempner partition):

```bash
python wam/submit.py "cd $(pwd) && python wam/scripts/train_wm_pusht_zarr.py \
  --zarr_path dataset/world_model_data/dataset_domain/all_data/domain18.zarr \
  --svd_model_path wam/svd --clip_model_path wam/clip \
  --ckpt_path wam/ckpt/checkpoint-10000.pt \
  --output_dir wam/ckpt/pusht_finetuned \
  --num_samples 20000 --max_train_steps 10000 --checkpointing_steps 2000 \
  --mixed_precision fp16 --no_wandb"
```

After training, switch the config checkpoint path:

```yaml
# gpc_wam_evaluation/configs/gpc_wam_evaluation_config.yml
wam:
  ckpt_path: ./wam/ckpt/pusht_finetuned/checkpoint-10000.pt
```

### IWS (interactive_world_sim LatentWorldModel)

`interactive_world_sim/` is a separate repo containing a **3-stage latent world model** that can be trained on PushT zarr data and used as an alternative scoring backend in both `gpc_wam_evaluation` and `gpc_rank_evaluation`.

#### Architecture

- **Stage 1** — CNN encoder + CM-decoder (consistency-model diffusion decoder): learns a compact latent representation of 96×96 PushT images. Encoder: stride-2 conv × 2 → `(4, 24, 24)` latents. Decoder: diffusion-based reconstruction from latents.
- **Stage 2** — `CMLatentDynamics` (causal UNet with temporal attention, action-conditioned via FiLM): trained to predict future latent states given past latents + actions. Encoder/decoder weights are frozen.
- **Stage 3** (optional) — Decoder fine-tuning with perturbed latents for robustness.

The `LatentWorldModel` is a PyTorch Lightning module trained via **Hydra** configuration. Hydra saves `outputs/<date>/<time>/.hydra/config.yaml` alongside checkpoints; the IWS adapter auto-detects this config from the checkpoint path.

The `LinearNormalizer` (images `[0,1]→[-1,1]`, actions `[0,511]→[-1,1]`) is an `nn.Module` saved inside the Lightning checkpoint — it is restored automatically when loading from a checkpoint.

#### New files added

| File | Purpose |
|------|---------|
| `interactive_world_sim/interactive_world_sim/datasets/latent_dynamics/pusht_zarr_dataset.py` | `PushTZarrDataset` — reads PushT zarr via `ReplayBuffer`, emits `obs: {image: (T,3,H,W)}` + `action: (T,2)` |
| `interactive_world_sim/configurations/dataset/pusht_dataset.yaml` | Hydra dataset config: `domain18.zarr`, `resolution=96`, `action_dim=2`, `obs_keys=[image]` |
| `gpc_wam_evaluation/iws_adapter.py` | `IWSWorldModelAdapter.rollout_final_image()` — same interface as WAM/GPC adapters |
| `gpc_rank_evaluation/eval_iws.py` | GPC-RANK eval loop using IWS as world model (same reward predictors as `eval_baseline.py`) |
| `gpc_rank_evaluation/gpc_rank_evaluation_iws.py` | CLI entry point for IWS-based GPC-RANK eval |
| `gpc_rank_evaluation/configs/gpc_rank_evaluation_iws_config.yml` | Config for IWS GPC-RANK eval |

#### Training IWS on PushT

Run from `interactive_world_sim/`. The `wandb.entity` flag is required by Hydra even when `wandb.mode=disabled` — pass any string:

```bash
cd interactive_world_sim

# Stage 1: train encoder + decoder (~200k steps)
python main.py \
  experiment=exp_latent_dyn \
  dataset=pusht_dataset \
  algorithm=latent_world_model \
  algorithm.training_stage=1 \
  algorithm.action_dim=2 \
  wandb.entity=dummy \
  wandb.mode=disabled \
  +name=pusht_stage1

# Stage 2: train dynamics, loading Stage 1 checkpoint (find path under outputs/)
python main.py \
  experiment=exp_latent_dyn \
  dataset=pusht_dataset \
  algorithm=latent_world_model \
  algorithm.training_stage=2 \
  algorithm.action_dim=2 \
  "algorithm.load_ae=outputs/<date>/<time>/checkpoints/<step>.ckpt" \
  wandb.entity=dummy \
  wandb.mode=disabled \
  +name=pusht_stage2
```

Checkpoints are saved to `interactive_world_sim/outputs/<date>/<time>/checkpoints/`. The Hydra config is at `outputs/<date>/<time>/.hydra/config.yaml` — the IWS adapter auto-detects it.

#### Evaluating with IWS as world model

**Via `gpc_wam_evaluation`** (MSE-to-goal-image reward):

```bash
# Online env, diffusion planner
python -m gpc_wam_evaluation.gpc_wam_evaluation \
  --world_model_type iws \
  --iws_ckpt_path interactive_world_sim/outputs/<date>/<time>/checkpoints/<step>.ckpt \
  --planner_mode diffusion \
  --num_episodes 100

# Offline sanity check (no env needed)
python -m gpc_wam_evaluation.gpc_wam_evaluation \
  --world_model_type iws \
  --iws_ckpt_path interactive_world_sim/outputs/<date>/<time>/checkpoints/<step>.ckpt \
  --eval_mode offline_dataset \
  --offline_zarr_path dataset/world_model_data/dataset_domain/all_data/domain18.zarr \
  --num_episodes 20
```

Config keys under `iws_world_model:` in YAML:
- `ckpt_path`: path to the Stage 2 Lightning checkpoint
- `cfg_path`: path to `.hydra/config.yaml` (auto-detected from `ckpt_path` if blank)
- `num_history`: number of context frames fed to the encoder (default `1`)
- `num_frames`: future steps to roll out per candidate (default `8`)

**Via `gpc_rank_evaluation`** (geometry-based reward predictors):

```bash
python -m gpc_rank_evaluation.gpc_rank_evaluation_iws \
  --iws_ckpt_path interactive_world_sim/outputs/<date>/<time>/checkpoints/<step>.ckpt
```

#### gpc_wam_evaluation internals

| File | Purpose |
|------|---------|
| `gpc_wam_evaluation.py` | CLI entry point; `--world_model_type gpc\|wam\|iws`; resolves checkpoints; dispatches to `eval_wam` |
| `eval_wam.py` | Main loop — online env or offline zarr; ranks candidates via `-MSE(pred, goal)` |
| `gpc_world_model_adapter.py` | `GpcWorldModelAdapter.rollout_final_image()` → `(3,H,W)` via autoregressive denoiser rollout |
| `wam_adapter.py` | `WamWorldModelAdapter.rollout_final_image()` → `(3,H,W)` via SVD video diffusion |
| `iws_adapter.py` | `IWSWorldModelAdapter.rollout_final_image()` → `(3,H,W)` via IWS encoder + dynamics + CM-decoder |
| `planners.py` | `diffusion_sample_actions` (uses GPC policy) and `mppi_sample_actions` (gradient-free optimization) |
| `configs/gpc_wam_evaluation_config.yml` | All knobs: `world_model_type`, `planner_mode`, `eval.mode`, `gpc_world_model.*`, `wam.*`, `iws_world_model.*`, `diffusion_policy.*`, `mppi.*` |

Reward signal: `reward = -MSE(predicted_final_image, goal_image)`. The goal image is obtained from `env._render_frame_target_only()` (online) or the episode's last frame (offline).

#### World model backends

**GPC backend** (`world_model_type: gpc`, default):
- Uses `all_checkpoints/world_model_checkpoint/phase_two_checkpoint/denoiser.pth`
- Single-step diffusion denoiser rolled out autoregressively: 4 context frames + 4 actions → 1 next frame, repeated for `num_rollout_steps` (= `action_horizon`, default 8)
- Operates at env resolution (96×96); no separate download required

**WAM backend** (`world_model_type: wam`):
- Uses Ctrl-World SVD model from `wam/ckpt/`; DROID pre-trained by default, fine-tune on PushT for better results
- See the WAM section above for training and checkpoint details

**IWS backend** (`world_model_type: iws`):
- Uses the `LatentWorldModel` from `interactive_world_sim/`, trained in two stages on PushT zarr
- Encodes history frames to latents → rolls out future latents via dynamics model → decodes final latent to pixel space via CM-decoder
- Requires a Stage 2 Lightning checkpoint (`.ckpt`) from the IWS training run

## Key Implementation Details

### Training Flow

1. **Data Loading**: Zarr-based datasets are loaded with trajectory normalization (min/max statistics per dimension)
2. **Model Architecture**: Uses denoiser-based diffusion with UNet backbone, configurable depths and attention layers
3. **Loss Computation**: Training loops through epochs with batch-level loss updates and periodic checkpoint saving
4. **Logging**: Optional W&B integration for experiment tracking

### Evaluation Flow

1. **Policy Generation**: Diffusion model samples `num_diffusion_iters` candidate action sequences
2. **World Model Scoring**: World model predicts future states for each trajectory
3. **Ranking/Optimization**: GPC-RANK selects highest-scoring trajectory; GPC-OPT refines via gradient descent
4. **Execution**: Execute selected trajectory in environment and collect video/statistics

### Dataset Format

- Zarr datasets with `image`, `agent_pos`, `action` keys
- Statistics stored per domain for normalization (min/max scaling)
- Configurable horizon windows (obs_horizon, pred_horizon, action_horizon)

## Debugging Tips

- Set `wandb: false` in configs to disable W&B logging during development
- Check `output_dir` / `models_save_dir` in configs to locate checkpoints and evaluation outputs
- Use `num_episodes: 1` and `max_steps: 300` for quick evaluation tests
- Inspect `eval_baseline.py` for the main evaluation loop and environment interaction logic
- **GPC-WAM**: use `eval.mode: offline_dataset` with a small `num_samples` for a fast end-to-end smoke test before running online eval
- **GPC-WAM**: pass `--no_wandb` to `train_wm_pusht_zarr.py` to skip W&B when the account is not configured
- **GPC-WAM**: `wam_adapter.py` safety-checks that `wam.*` imports resolve to the local symlink, not a system-installed package; always run from repo root
- **GPC-WAM**: the WAM checkpoint (`checkpoint-10000.pt` in `wam/ckpt/`) is the DROID pre-trained model; for best PushT results, fine-tune it first using `train_wm_pusht_zarr.py`
- **GPC-WAM (GPC backend)**: `gpc_world_model_adapter.py` adds `gpc_rank_evaluation/` to `sys.path` at import time — do not remove this; the denoiser and inner_model use bare absolute imports (`data`, `diffusion`) that only resolve when that directory is on the path
- **GPC-WAM (GPC backend)**: the YAML `offline_dataset:` section has all children commented out, so PyYAML parses it as `None`; always use `config.get("offline_dataset") or {}` (not `setdefault`) when reading sub-keys
- **skvideo / NumPy 2.0**: `ndarray.tostring()` was removed in NumPy 2.0; `_maybe_vwrite` in `eval_wam.py` patches `np.ndarray.tostring = np.ndarray.tobytes` at call time as a compatibility shim
- **IWS training (Hydra / wandb)**: `main.py` validates `wandb.entity` even when `wandb.mode=disabled`; always pass `wandb.entity=dummy` when disabling wandb on the command line
- **IWS training (Hydra CWD)**: `@hydra.main(version_base=None)` changes the working directory to `outputs/<date>/<time>/` before your code runs. `PushTZarrDataset` uses `hydra.utils.get_original_cwd()` to resolve the relative `dataset_path` back to the original `interactive_world_sim/` launch directory. The default config uses `../dataset/...` so it resolves correctly to the repo-root zarr. If you move or rename the zarr, pass an absolute path via `dataset.dataset_path=/abs/path/to/domain18.zarr`
- **IWS adapter (auto-detect config)**: `iws_adapter.py` walks up the checkpoint path looking for `.hydra/config.yaml`; if the checkpoint was copied out of its Hydra output tree, pass `--iws_cfg_path` explicitly
- **IWS normalizer**: the `LinearNormalizer` is an `nn.Module` saved inside the Lightning checkpoint; if a checkpoint was saved before `set_normalizer()` was called (e.g., a pre-training stub), the adapter falls back to a hard-coded PushT action range `[0, 511]`
- **IWS `action_dim`**: must be overridden to `2` on the command line (`algorithm.action_dim=2`) since the default config assumes ALOHA's 10-DoF actions

## Multimodal World Model (`world_model_multimodal/`)

Three fusion strategies for training a world model on the ManiFEEL robot manipulation dataset. All three share the same two input modalities:
- **Visual**: `front` RGB camera (256×256 → resized to 96×96)
- **Tactile**: `left_tactile_camera_taxim` camera (320×240 → resized to 96×96)
- **Actions**: 7-DoF continuous robot actions, normalized to [-1, 1] per dataset

**Dataset path**: `/n/holylabs/ydu_lab/Lab/pwu/Projects/3d_video_model/data/manifeel_data`

Zarr structure (one store per task under `manifeel_data/data/<task>/`):
- Arrays live under the `data/` group: `data/front`, `data/left_tactile_camera_taxim`, `data/action`
- Episode boundaries at `meta/episode_ends` (exclusive end indices)
- Tasks: `nutbolt_quan_July1`, `bulb_quan_Sep19`, `gear_quan_Sep15`, `pih_quan_June06`, `plug_quan_Aug02`, `usb_quan_Aug05`, `blindinsert_quan_Aug15`, `sorting_quan_Aug8`, `explore_quan_June17`

### Training Commands

All three scripts must be run from **within their respective directories** (configs use relative paths):

```bash
# Early Fusion — concatenate RGB + tactile as 6-channel input before the first conv
cd world_model_multimodal/early_fusion
python train.py --config configs/config.yml

# Middle Fusion — dual UNet encoders + cross-modal attention at the bottleneck
cd world_model_multimodal/middle_fusion
python train.py --config configs/config.yml

# Late Fusion — two independent experts (RGB and tactile), scores composed at inference
cd world_model_multimodal/late_fusion
python train.py --config configs/config.yml
```

Checkpoints are saved to `saved_checkpoints/checkpoint_epoch_N/` inside each fusion directory.

### Architecture Summary

| Fusion type | Directory | Key idea | Output |
|---|---|---|---|
| **Early** | `early_fusion/` | `cat(front, tactile)` → 6-ch input; one UNet sees both modalities identically | 6-ch (RGB + tactile next frame) |
| **Middle** | `middle_fusion/` | Separate RGB and tactile encoder streams; `CrossModalAttention` at bottleneck; shared decoder | 3-ch (next RGB frame) |
| **Late** | `late_fusion/` | `model_rgb` + `model_tac` each predict noise for the same RGB target; `eps = w_rgb·ε_rgb + w_tac·ε_tac` at inference | 3-ch (next RGB frame) |

### Key Config Knobs

| Key | Default | Notes |
|---|---|---|
| `obs_horizon` | 4 | Past frames used as conditioning context |
| `pred_horizon` | 8 | Autoregressive future steps per training window |
| `resize_scale` | 96 | Spatial resolution for both modalities |
| `batch_size` | 8 | |
| `lr` | 1e-4 | AdamW |
| `wandb` | false | Set to true to enable W&B logging |
| `loss_alpha` | 0.5 | Late fusion only: `alpha·loss_rgb + (1-alpha)·loss_tac` |
| `composition_weights` | [0.7, 0.3] | Late fusion only: `[w_rgb, w_tac]` at inference |

### Debugging Tips

- **Early fusion**: `img_channels=6` throughout; `conv_out` predicts both RGB and tactile next frame. MSE loss is computed over all 6 channels.
- **Middle fusion**: tactile frames are **not** noise-augmented during training (only the RGB conditioning window is); `CrossModalAttention` is zero-initialized so it starts as identity — training is stable from step 1.
- **Late fusion**: the RGB expert's denoised output becomes the rolling `prev_rgb` for subsequent autoregressive steps; tactile is always read from ground truth (never predicted). Score composition `w_rgb * out_rgb + w_tac * out_tac` is applied to raw model outputs **before** the `c_skip/c_out` EDM wrapping — this is the mathematically correct composition point.
- **DataLoader workers**: middle and late fusion keep zarr arrays as lazy references (not pre-loaded into RAM). This is intentional since multi-task images would exceed RAM. Zarr supports concurrent reads with `num_workers=4`.
- **Early fusion loads into RAM**: all frames are pre-loaded at dataset init (safer for DataLoader workers but requires more memory per task).
