"""GPC-RANK evaluation using the interactive_world_sim LatentWorldModel.

Replaces the Denoiser-based world model in eval_baseline.py with the IWS
LatentWorldModel (encoder + dynamics + CM-decoder pipeline).  The action
policy, environment, and reward predictors are identical to eval_baseline.py.

Typical entry point
-------------------
    python -m gpc_rank_evaluation.gpc_rank_evaluation_iws \
        --config gpc_rank_evaluation/configs/gpc_rank_evaluation_iws_config.yml

Or call eval_iws(config, policy_checkpoint) directly.
"""

from __future__ import annotations

import collections
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.transforms import v2
from tqdm.auto import tqdm

from .utils import (
    normalize_data,
    unnormalize_data,
    create_injected_noise,
)
from .pusht_env import PushTImageEnv
from .models import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from .ema import SimpleEMAModel
from .eval_baseline import RewardPredictor, estimate_reward_torch, transform_vertices_torch

# ---------------------------------------------------------------------------
# IWS world model (imported from gpc_wam_evaluation adapter)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).absolute().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpc_wam_evaluation.iws_adapter import IWSAdapterConfig, IWSWorldModelAdapter

# ---------------------------------------------------------------------------
# Constants that match eval_baseline.py
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DOMAIN18_STATS = {
    "agent_pos": {
        "min": np.array([9.897889, 9.63592], dtype=np.float32),
        "max": np.array([499.517, 499.00488], dtype=np.float32),
    },
    "action": {
        "min": np.array([2.0, 2.0], dtype=np.float32),
        "max": np.array([511.0, 511.0], dtype=np.float32),
    },
}


def _maybe_vwrite(path: str, frames, enabled: bool) -> None:
    if not enabled:
        return
    try:
        from skvideo.io import vwrite  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'skvideo'. Set save_video: false in config to skip."
        ) from e
    vwrite(path, frames)


def _load_diffusion_policy(models_save_dir: str, obs_horizon: int):
    """Load the GPC diffusion action policy (identical to eval_baseline.py)."""
    nets = nn.ModuleDict({})
    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets["vision_encoder"] = vision_encoder

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    nets["invariant"] = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
    )
    nets = nets.to(device)

    ema = SimpleEMAModel.from_parameters(nets.parameters(), power=0.75)
    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    ema_path = os.path.join(models_save_dir, "ema_nets.pth")
    ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
    ema.copy_to(nets.parameters())
    return nets


def eval_iws(config: dict, models_save_dir: str) -> list:
    """GPC-RANK evaluation loop using the IWS LatentWorldModel.

    Parameters
    ----------
    config : dict
        Configuration dict (see gpc_rank_evaluation_iws_config.yml).
    models_save_dir : str
        Directory containing the diffusion policy checkpoints
        (vision_encoder.pth, invariant.pth, ema_nets.pth).
    """
    num_diffusion_iters = config["num_diffusion_iters"]
    pred_horizon = config["pred_horizon"]
    obs_horizon = config["obs_horizon"]
    action_horizon = config["action_horizon"]
    output_dir = config["output_dir"]
    resize_scale = config["resize_scale"]
    num_trial = config.get("num_trial", 50)
    save_video = bool(config.get("save_video", False))
    start_number_test = int(config.get("start_number_test", 0))
    num_episodes = int(config.get("num_episodes", 100))
    max_steps = config["max_steps"]

    os.makedirs(output_dir, exist_ok=True)

    # --- Build IWS world model ---
    iws_cfg = config["iws_world_model"]
    adapter_cfg = IWSAdapterConfig(
        ckpt_path=str(iws_cfg["ckpt_path"]),
        cfg_path=str(iws_cfg.get("cfg_path", "")),
        num_history=int(iws_cfg.get("num_history", 1)),
        num_frames=int(iws_cfg.get("num_frames", action_horizon)),
    )
    world_model = IWSWorldModelAdapter(adapter_cfg, device=device)
    wm_num_history = adapter_cfg.num_history
    wm_num_frames = adapter_cfg.num_frames
    print(f"[eval_iws] IWS world model loaded (num_history={wm_num_history}, num_frames={wm_num_frames})")

    # --- Load diffusion action policy ---
    ema_nets = _load_diffusion_policy(models_save_dir, obs_horizon)
    ema_nets.eval()

    # --- Load reward predictors ---
    reward_predictor_xy = RewardPredictor().to(device)
    reward_predictor_angle = RewardPredictor().to(device)
    reward_predictor_xy.load_state_dict(
        torch.load(config["reward_predictor_xy_checkpoint"], map_location="cpu")
    )
    reward_predictor_angle.load_state_dict(
        torch.load(config["reward_predictor_angle_checkpoint"], map_location="cpu")
    )
    reward_predictor_xy.eval()
    reward_predictor_angle.eval()
    print("[eval_iws] All models loaded.")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(96),
        v2.ToDtype(torch.float32, scale=True),
    ])

    scores = []
    env_j_scores = []
    env_seed = 100000 + start_number_test

    with open("./domains_yaml/push_t.yml", "r") as f:
        data_loaded = yaml.safe_load(f)
    env_id = data_loaded["domain_id"]

    print(f"\n[eval_iws] Eval IWS on Domain #{env_id}:")

    end_number_test = start_number_test + num_episodes

    for test_index in range(start_number_test, end_number_test):
        noise_scheduler = create_injected_noise(num_diffusion_iters)

        env = PushTImageEnv(domain_filename="push_t", resize_scale=resize_scale)
        env.seed(env_seed)
        obs, info = env.reset()

        target_pose = torch.tensor(env.goal_pose, dtype=torch.float32).to(device)
        target_pose[2] = target_pose[2] % (2 * np.pi)

        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        # History buffer for IWS world model (stores actual frames as CHW float [0,1])
        wm_hist_deque = collections.deque(
            [obs["image"]] * wm_num_history, maxlen=wm_num_history
        )

        imgs = [env.render(mode="rgb_array")]
        rewards = []
        done = False
        step_idx = 0
        all_action_save: list = []
        tqdm._instances.clear()

        with tqdm(total=max_steps, desc=f"Eval Trial #{test_index}") as pbar:
            while not done:
                # --- Stack current observation ---
                images = np.stack([x["image"] for x in obs_deque])       # (obs_horizon,3,H,W)
                agent_poses = np.stack([x["agent_pos"] for x in obs_deque])
                nagent_poses = normalize_data(agent_poses, stats=_DOMAIN18_STATS["agent_pos"])

                nimages = torch.from_numpy(images).to(device, dtype=torch.float32)
                nagent_poses_t = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                # --- Sample action candidates via diffusion policy ---
                with torch.no_grad():
                    image_features = ema_nets["vision_encoder"](nimages).squeeze()
                    obs_features = torch.cat([image_features, nagent_poses_t], dim=-1)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    noisy_action = torch.randn((num_trial, pred_horizon, 2), device=device)
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                    for k in noise_scheduler.timesteps:
                        noise_pred = ema_nets["invariant"](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond.repeat(num_trial, 1),
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction,
                        ).prev_sample

                    naction = naction.detach().cpu().numpy()
                    action_pred = unnormalize_data(naction, stats=_DOMAIN18_STATS["action"])

                    start = obs_horizon - 1
                    end = start + action_horizon
                    action_candidates = action_pred[:, start:end, :]    # (num_trial, action_horizon, 2)

                    # Spread candidates slightly around their mean (matches eval_baseline.py)
                    action_mean = np.mean(action_candidates, axis=0, keepdims=True)
                    action_mean = np.repeat(action_mean, num_trial, axis=0)
                    action_candidates = action_mean + 1.01 * (action_candidates - action_mean)

                # --- Score candidates via IWS world model ---
                wm_hist = np.stack(list(wm_hist_deque))  # (wm_num_history, 3, H, W)

                all_reward_candidate = []
                for i in range(num_trial):
                    act = action_candidates[i][:wm_num_frames]  # (wm_num_frames, 2)
                    with torch.no_grad():
                        pred_final = world_model.rollout_final_image(wm_hist, act)
                        # pred_final: (3, H, W) float32 [0,1]

                    pred_final_t = pred_final.unsqueeze(0).to(device)  # (1,3,H,W)
                    unnorm_xy = reward_predictor_xy(pred_final_t)[0]
                    cossin = reward_predictor_angle(pred_final_t)[0]
                    cossin = cossin / (torch.norm(cossin) + 1e-8)
                    angle = torch.atan2(cossin[1], cossin[0]) % (2 * torch.pi)
                    block_pose = torch.stack([unnorm_xy[0], unnorm_xy[1], angle])
                    r = estimate_reward_torch(block_pose, target_pose)
                    all_reward_candidate.append(r.detach().cpu().item())

                best_idx = int(np.argsort(all_reward_candidate)[0])
                action_pick = action_candidates[best_idx][:end]

                # --- Execute best action in environment ---
                for i in range(len(action_pick)):
                    obs, reward, done, _, info = env.step(action_pick[i])
                    obs_deque.append(obs)
                    wm_hist_deque.append(obs["image"])
                    rewards.append(reward)
                    imgs.append(env.render(mode="rgb_array"))
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix({"current": reward, "max": max(rewards)})
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        env_seed += 1
        env_j_scores.append(max(rewards))
        if save_video:
            _maybe_vwrite(
                os.path.join(output_dir, f"iws_domain_{env_id}_test_{test_index}.mp4"),
                imgs,
                enabled=True,
            )
        np.save(
            os.path.join(
                output_dir,
                f"iws_scores_from_index_{start_number_test}.npy",
            ),
            np.array(env_j_scores),
        )

    print(f"[eval_iws] Domain #{env_id} Avg Score: {np.mean(env_j_scores):.4f}")
    scores.append(env_j_scores)
    np.save(
        os.path.join(output_dir, f"iws_scores_from_index_{start_number_test}.npy"),
        np.array(scores),
    )
    return scores
