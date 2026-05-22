from __future__ import annotations

import os
import collections
from typing import Dict, Any

import numpy as np
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'torch'. Please activate the project environment (see environment.yml) "
        "before running gpc_wam_evaluation."
    ) from e
from tqdm.auto import tqdm
def _maybe_vwrite(path: str, frames, enabled: bool) -> None:
    if not enabled or not frames:
        return
    import imageio
    writer = imageio.get_writer(path, fps=10, codec="libx264", quality=None, macro_block_size=None, pixelformat="yuv420p")
    try:
        for frame in frames:
            # env.render returns RGB HWC uint8 — imageio expects the same
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _save_goal_image(path: str, goal_chw_01: np.ndarray) -> None:
    """Save CHW float [0,1] goal image as a PNG file."""
    import imageio
    hwc_uint8 = (np.moveaxis(goal_chw_01, 0, -1) * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite(path, hwc_uint8)

from gpc_rank_evaluation.utils import (
    create_injected_noise,
    normalize_data,
    unnormalize_data,
)
from gpc_rank_evaluation.models import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from gpc_rank_evaluation.pusht_env import PushTImageEnv
from gpc_rank_evaluation.ema import SimpleEMAModel

from .planners import diffusion_sample_actions, mppi_sample_actions
from .wam_adapter import WamAdapterConfig, WamWorldModelAdapter
from .gpc_world_model_adapter import GpcWorldModelConfig, GpcWorldModelAdapter


def _to_torch_image(chw: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32)


def _goal_state_image(env: PushTImageEnv) -> np.ndarray:
    """
    Render the scene with the T-block placed at the goal pose.
    This is what a perfectly solved episode looks like: red T covering the black
    goal outline. Using _render_frame_target_only (goal outline only, no T) was
    wrong because the world model always predicts the full scene including the red
    T, so -MSE against a target-only image is never small even for a perfect match.
    """
    old_pos = env.block.position
    old_angle = env.block.angle
    env.block.position = env.goal_pose[:2].tolist()
    env.block.angle = float(env.goal_pose[2])
    img = env._render_frame(mode="rgb_array")
    env.block.position = old_pos
    env.block.angle = old_angle
    img = img.astype(np.float32) / 255.0
    return np.moveaxis(img, -1, 0)


def _mse_reward(pred_final_chw: torch.Tensor, goal_chw: torch.Tensor) -> float:
    # reward = -MSE
    return -torch.mean((pred_final_chw - goal_chw) ** 2).item()


def _hwc_uint8_to_chw_float01(img_hwc: np.ndarray) -> np.ndarray:
    img = img_hwc.astype(np.float32) / 255.0
    return np.moveaxis(img, -1, 0)


def _resize_chw_np(chw: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W)
    t = torch.nn.functional.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).cpu().numpy()


def _build_world_model(config: Dict[str, Any], device: torch.device):
    """
    Returns (world_model, wm_num_history, wam_cfg_or_none).
    wm_num_history: number of context frames the model needs.
    wam_cfg_or_none: WamAdapterConfig if wm_type=="wam", else None.
    """
    wm_type = config.get("world_model_type", "wam")

    if wm_type == "wam":
        wam_cfg_raw = config["wam"]
        wam_cfg = WamAdapterConfig(
            ckpt_path=wam_cfg_raw["ckpt_path"],
            svd_model_path=wam_cfg_raw["svd_model_path"],
            clip_model_path=wam_cfg_raw["clip_model_path"],
            num_history=int(wam_cfg_raw["num_history"]),
            num_frames=int(wam_cfg_raw["num_frames"]),
            action_dim=int(wam_cfg_raw["action_dim"]),
            width=int(wam_cfg_raw["width"]),
            height=int(wam_cfg_raw["height"]),
            fps=int(wam_cfg_raw["fps"]),
            motion_bucket_id=int(wam_cfg_raw["motion_bucket_id"]),
            guidance_scale=float(wam_cfg_raw["guidance_scale"]),
            num_inference_steps=int(wam_cfg_raw["num_inference_steps"]),
            decode_chunk_size=int(wam_cfg_raw["decode_chunk_size"]),
            dtype=str(wam_cfg_raw.get("dtype", "bf16")),
        )
        world_model = WamWorldModelAdapter(wam_cfg, device=device)
        return world_model, wam_cfg.num_history, wam_cfg
    elif wm_type == "gpc":
        gpc_cfg_raw = config["gpc_world_model"]
        gpc_cfg = GpcWorldModelConfig(
            ckpt_path=str(gpc_cfg_raw["ckpt_path"]),
            num_diffusion_steps=int(gpc_cfg_raw.get("num_diffusion_steps", 3)),
        )
        world_model = GpcWorldModelAdapter(gpc_cfg, device=device)
        return world_model, GpcWorldModelAdapter.num_history, None
    elif wm_type == "iws":
        from .iws_adapter import IWSAdapterConfig, IWSWorldModelAdapter
        iws_cfg_raw = config["iws_world_model"]
        iws_cfg = IWSAdapterConfig(
            ckpt_path=str(iws_cfg_raw["ckpt_path"]),
            cfg_path=str(iws_cfg_raw.get("cfg_path", "")),
            num_history=int(iws_cfg_raw.get("num_history", 1)),
            num_frames=int(iws_cfg_raw.get("num_frames", 8)),
        )
        world_model = IWSWorldModelAdapter(iws_cfg, device=device)
        return world_model, iws_cfg.num_history, None
    else:
        raise ValueError(f"Unknown world_model_type={wm_type!r}. Expected 'wam', 'gpc', or 'iws'.")


def _prepare_goal_image_t(
    goal_chw_01: np.ndarray,
    wm_type: str,
    wam_cfg,  # WamAdapterConfig or None
    device: torch.device,
) -> torch.Tensor:
    """Resize goal image to the resolution the world model outputs, then move to device."""
    if wm_type == "wam" and wam_cfg is not None:
        if goal_chw_01.shape[-2:] != (wam_cfg.height, wam_cfg.width):
            goal_chw_01 = _resize_chw_np(goal_chw_01, (wam_cfg.height, wam_cfg.width))
    # GPC operates at env resolution (96×96), no resize needed.
    return _to_torch_image(goal_chw_01, device)


def eval_wam_offline_dataset(config: Dict[str, Any]) -> np.ndarray:
    """
    Offline evaluation from a zarr dataset:
    - sample episodes
    - use episode-final image as goal image
    - score a fixed trajectory segment by reward = -MSE(pred_final_img, goal_img)
    """
    import zarr

    seed = int(config.get("seed", 1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_type = config.get("world_model_type", "wam")

    ev_cfg = config["eval"]
    world_model, wm_num_history, wam_cfg = _build_world_model(config, device)

    # Number of future action steps to roll out per sample.
    if wm_type == "wam":
        wm_num_frames = wam_cfg.num_frames
    elif wm_type == "iws":
        wm_num_frames = int(config["iws_world_model"].get("num_frames", ev_cfg.get("action_horizon", 8)))
    else:
        gpc_cfg_raw = config["gpc_world_model"]
        wm_num_frames = int(gpc_cfg_raw.get("num_rollout_steps", ev_cfg.get("action_horizon", 8)))

    offline_cfg = ev_cfg.get("offline_dataset") or {}
    zarr_path = str(offline_cfg.get("zarr_path", "dataset/world_model_data/dataset_domain/all_data/domain18.zarr"))
    num_samples = int(offline_cfg.get("num_samples", ev_cfg.get("num_episodes", 100)))
    use_dataset_actions = bool(offline_cfg.get("use_dataset_actions", True))
    start_mode = str(offline_cfg.get("start_mode", "random"))  # random | episode_start

    output_dir = ev_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"[gpc_wam_eval] mode=offline_dataset world_model={wm_type} output_dir={output_dir}", flush=True)

    z = zarr.open(zarr_path, mode="r")
    imgs = z["data"]["img"]  # (N,H,W,3) uint8
    actions = z["data"]["action"]  # (N,2) float32
    episode_ends = np.asarray(z["meta"]["episode_ends"][:], dtype=np.int64)
    num_eps = int(len(episode_ends))
    if num_eps <= 0:
        raise ValueError(f"No episodes found in zarr meta.episode_ends at {zarr_path}")

    episode_starts = np.zeros_like(episode_ends)
    episode_starts[0] = 0
    episode_starts[1:] = episode_ends[:-1]

    scores = []
    chosen = []
    for idx in tqdm(range(num_samples), desc="Offline eval (dataset)"):
        ep = int(rng.integers(0, num_eps))
        s = int(episode_starts[ep])
        e = int(episode_ends[ep])
        length = e - s
        if length <= (wm_num_history + wm_num_frames):
            continue

        if start_mode == "episode_start":
            t0 = s + wm_num_history
        else:
            t0 = int(rng.integers(s + wm_num_history, e - wm_num_frames))

        hist_idx = np.arange(t0 - wm_num_history, t0, dtype=np.int64)
        act_idx = np.arange(t0, t0 + wm_num_frames, dtype=np.int64)

        hist_imgs = np.stack([_hwc_uint8_to_chw_float01(imgs[i]) for i in hist_idx], axis=0)

        # WAM resizes internally; GPC operates at env resolution — no resize needed.
        if wm_type == "wam" and hist_imgs.shape[-2:] != (wam_cfg.height, wam_cfg.width):
            hist_imgs = np.stack(
                [_resize_chw_np(hist_imgs[i], (wam_cfg.height, wam_cfg.width)) for i in range(hist_imgs.shape[0])],
                axis=0,
            )

        if use_dataset_actions:
            act2 = np.asarray(actions[act_idx], dtype=np.float32)  # (wm_num_frames, 2)
        else:
            raise ValueError("offline_dataset.use_dataset_actions=false is not implemented.")

        goal = _hwc_uint8_to_chw_float01(np.asarray(imgs[e - 1]))
        goal_t = _prepare_goal_image_t(goal, wm_type, wam_cfg, device)

        pred_final = world_model.rollout_final_image(hist_imgs, act2)
        score = _mse_reward(pred_final.to(device), goal_t)
        scores.append(float(score))
        chosen.append((ep, int(t0)))
        if (idx + 1) % 10 == 0:
            print(f"[gpc_wam_eval] offline progress {idx + 1}/{num_samples}", flush=True)

    scores_arr = np.asarray(scores, dtype=np.float32)
    np.save(os.path.join(output_dir, "offline_scores.npy"), scores_arr)
    np.save(os.path.join(output_dir, "offline_chosen_ep_t0.npy"), np.asarray(chosen, dtype=np.int64))
    print(f"[gpc_wam_eval] wrote {os.path.join(output_dir, 'offline_scores.npy')}", flush=True)
    return scores_arr


def _load_diffusion_policy(policy_checkpoint: str, obs_horizon: int, device: torch.device):
    nets = nn.ModuleDict({})
    vision_encoder = replace_bn_with_gn(get_resnet())
    nets["vision_encoder"] = vision_encoder

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2
    nets["invariant"] = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon)

    nets = nets.to(device)
    ema = SimpleEMAModel.from_parameters(nets.parameters(), power=0.75)

    for model_name, model in nets.items():
        model_path = os.path.join(policy_checkpoint, f"{model_name}.pth")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    ema_path = os.path.join(policy_checkpoint, "ema_nets.pth")
    ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
    ema.copy_to(nets.parameters())
    nets.eval()
    return nets


def eval_wam(config: Dict[str, Any]) -> np.ndarray:
    ev_cfg = config.get("eval", {})
    if ev_cfg.get("mode", "online_env") == "offline_dataset":
        return eval_wam_offline_dataset(config)

    seed = int(config.get("seed", 1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_type = config.get("world_model_type", "wam")

    env_cfg = config["env"]
    ev_cfg = config["eval"]

    obs_horizon = int(ev_cfg["obs_horizon"])
    pred_horizon = int(ev_cfg["pred_horizon"])
    action_horizon = int(ev_cfg["action_horizon"])
    max_steps = int(env_cfg["max_steps"])
    resize_scale = int(env_cfg["resize_scale"])

    output_dir = ev_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"[gpc_wam_eval] mode=online_env world_model={wm_type} output_dir={output_dir}", flush=True)

    env = PushTImageEnv(domain_filename=env_cfg["domain_filename"], resize_scale=resize_scale)

    world_model, wm_num_history, wam_cfg = _build_world_model(config, device)
    print(f"[gpc_wam_eval] world model loaded", flush=True)

    # Number of action steps fed into the world model per scoring call.
    if wm_type == "wam":
        wm_num_frames = wam_cfg.num_frames
    elif wm_type == "iws":
        wm_num_frames = int(config["iws_world_model"].get("num_frames", action_horizon))
    else:
        wm_num_frames = action_horizon

    planner_mode = config["planner_mode"]
    if planner_mode not in {"diffusion", "mppi"}:
        raise ValueError(f"Unsupported planner_mode={planner_mode}, expected diffusion|mppi")

    ema_nets = None
    noise_scheduler = None
    if planner_mode == "diffusion":
        pol_cfg = config["diffusion_policy"]
        ema_nets = _load_diffusion_policy(pol_cfg["policy_checkpoint"], obs_horizon, device)
        noise_scheduler = create_injected_noise(int(pol_cfg["num_diffusion_iters"]))

    domain_stats = {
        "agent_pos": {"min": np.array([9.897889, 9.63592], dtype=np.float32), "max": np.array([499.517, 499.00488], dtype=np.float32)},
        "action": {"min": np.array([2.0, 2.0], dtype=np.float32), "max": np.array([511.0, 511.0], dtype=np.float32)},
    }

    num_episodes = int(ev_cfg["num_episodes"])
    save_video = bool(ev_cfg.get("save_video", True))
    episode_scores = []
    print(
        f"[gpc_wam_eval] planner={planner_mode} episodes={num_episodes} "
        f"max_steps={max_steps} save_video={save_video}",
        flush=True,
    )

    for ep in range(num_episodes):
        print(f"[gpc_wam_eval] episode {ep + 1}/{num_episodes} start", flush=True)
        env.seed(100000 + ep)
        obs, _info = env.reset()

        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

        # GPC / IWS need a dedicated history deque; WAM replicates the last frame.
        gpc_obs_deque = (
            collections.deque([obs] * wm_num_history, maxlen=wm_num_history)
            if wm_type in ("gpc", "iws") else None
        )

        # Prepare goal image at the resolution the world model compares at.
<<<<<<< HEAD
        goal_img_raw = _goal_state_image(env)  # (3, resize_scale, resize_scale) in [0,1]
        _save_goal_image(os.path.join(output_dir, f"goal_image_ep{ep:03d}.png"), goal_img_raw)
        # Save one (observation, goal) pair from the very first episode for inspection.
        if ep == 0:
            _save_goal_image(os.path.join(output_dir, "obs_image_ep000.png"), obs["image"])
            np.save(os.path.join(output_dir, "obs_ep000.npy"), obs["image"])
            np.save(os.path.join(output_dir, "goal_ep000.npy"), goal_img_raw)
            print(f"[gpc_wam_eval] saved (obs, goal) pair for ep0 to {output_dir}", flush=True)
=======
        goal_img_raw = _goal_only_image(env)  # (3, resize_scale, resize_scale) in [0,1]
        _save_goal_image(os.path.join(output_dir, f"goal_image_ep{ep:03d}.png"), goal_img_raw)
>>>>>>> 3158cf57c68130dfbd69117df9148db77174a670
        goal_img_t = _prepare_goal_image_t(goal_img_raw, wm_type, wam_cfg, device)

        rewards = []
        done = False
        step_idx = 0
        episode_imgs = [env.render(mode="rgb_array")] if save_video else None

        with tqdm(total=max_steps, desc=f"Eval ep {ep}", leave=False) as pbar:
            while not done and step_idx < max_steps:
                images = np.stack([x["image"] for x in obs_deque])  # (obs_horizon,3,H,W)
                agent_poses = np.stack([x["agent_pos"] for x in obs_deque])
                nagent_poses = normalize_data(agent_poses, stats=domain_stats["agent_pos"])

                if planner_mode == "diffusion":
                    nimages = torch.from_numpy(images).to(device=device, dtype=torch.float32)
                    nagent_poses_t = torch.from_numpy(nagent_poses).to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        image_features = ema_nets["vision_encoder"](nimages).squeeze()
                        obs_features = torch.cat([image_features, nagent_poses_t], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    nactions = diffusion_sample_actions(
                        ema_nets=ema_nets,
                        noise_scheduler=noise_scheduler,
                        obs_cond=obs_cond,
                        num_candidates=int(config["diffusion_policy"]["num_candidates"]),
                        num_diffusion_iters=int(config["diffusion_policy"]["num_diffusion_iters"]),
                        pred_horizon=pred_horizon,
                        action_dim=2,
                        device=device,
                    )
                    action_pred = unnormalize_data(nactions, stats=domain_stats["action"])
                    start = obs_horizon - 1
                    end = start + action_horizon
                    candidates = action_pred[:, start:end, :]  # (N, action_horizon, 2)
                else:
                    last_pos = agent_poses[-1].astype(np.float32)
                    mean = np.tile(last_pos[None, :], (action_horizon, 1))

                    def score_fn(cand: np.ndarray) -> np.ndarray:
                        # cand: (N, action_horizon, 2)
                        wm_hist = _get_wm_history(wm_type, images, gpc_obs_deque, wm_num_history)
                        out = []
                        for i in range(cand.shape[0]):
                            act = cand[i][: wm_num_frames]
                            pred_final = world_model.rollout_final_image(wm_hist, act)
                            out.append(_mse_reward(pred_final.to(device), goal_img_t))
                        return np.array(out, dtype=np.float32)

                    best, _scores = mppi_sample_actions(
                        mean_action=mean,
                        num_candidates=int(config["mppi"]["num_candidates"]),
                        num_iterations=int(config["mppi"]["num_iterations"]),
                        sigma=float(config["mppi"]["action_noise_sigma"]),
                        temperature=float(config["mppi"]["temperature"]),
                        clamp_min=float(config["mppi"]["clamp_action_min"]),
                        clamp_max=float(config["mppi"]["clamp_action_max"]),
                        score_fn=score_fn,
                        rng=rng,
                    )
                    candidates = best[None, :, :]

                # Rank candidates via world model MSE reward.
                wm_hist = _get_wm_history(wm_type, images, gpc_obs_deque, wm_num_history)
                scores = []
                for i in range(candidates.shape[0]):
                    act = candidates[i][: wm_num_frames]
                    pred_final = world_model.rollout_final_image(wm_hist, act)
                    scores.append(_mse_reward(pred_final.to(device), goal_img_t))
                best_idx = int(np.argmax(np.asarray(scores)))
                action_pick = candidates[best_idx]

                # Save the full predicted trajectory for the best candidate at the first
                # planning step of ep 0. Only supported for the GPC world model.
                if ep == 0 and step_idx == 0 and wm_type == "gpc":
                    traj = world_model.rollout_trajectory(wm_hist, action_pick[:wm_num_frames])
                    # traj: (T, 3, H, W) float32 [0,1]
                    np.save(os.path.join(output_dir, "wm_trajectory_ep000.npy"), traj)
                    traj_frames_hwc = [
                        (np.moveaxis(traj[i], 0, -1) * 255).clip(0, 255).astype(np.uint8)
                        for i in range(len(traj))
                    ]
                    _maybe_vwrite(os.path.join(output_dir, "wm_trajectory_ep000.mp4"), traj_frames_hwc, enabled=True)
                    print(
                        f"[gpc_wam_eval] saved world model trajectory ({len(traj)} frames) "
                        f"to {output_dir}/wm_trajectory_ep000.{{npy,mp4}}",
                        flush=True,
                    )

                for i in range(action_pick.shape[0]):
                    obs, reward, terminated, truncated, _info = env.step(action_pick[i])
                    done = bool(terminated or truncated)
                    obs_deque.append(obs)
                    if gpc_obs_deque is not None:
                        gpc_obs_deque.append(obs)
                    rewards.append(float(reward))
                    if save_video:
                        episode_imgs.append(env.render(mode="rgb_array"))
                    step_idx += 1
                    pbar.update(1)
                    if done or step_idx >= max_steps:
                        break

        episode_scores.append(max(rewards) if rewards else 0.0)
        print(f"[gpc_wam_eval] episode {ep + 1}/{num_episodes} done score={episode_scores[-1]:.4f}", flush=True)
        if save_video and episode_imgs is not None:
            _maybe_vwrite(os.path.join(output_dir, f"episode_{ep:03d}_{planner_mode}.mp4"), episode_imgs, enabled=save_video)

    episode_scores = np.asarray(episode_scores, dtype=np.float32)
    np.save(os.path.join(output_dir, f"scores_{planner_mode}.npy"), episode_scores)
    print(f"[gpc_wam_eval] wrote {os.path.join(output_dir, f'scores_{planner_mode}.npy')}", flush=True)
    return episode_scores


def _get_wm_history(
    wm_type: str,
    images: np.ndarray,            # (obs_horizon, 3, H, W) from obs_deque
    gpc_obs_deque,                  # deque of obs dicts, or None
    wm_num_history: int,
) -> np.ndarray:
    """
    Returns the history array to pass to world_model.rollout_final_image().

    WAM: repeats the latest frame wm_num_history times (WAM resizes internally).
    GPC: stacks the last wm_num_history actual observation frames.
    """
    if wm_type == "wam":
        return np.repeat(images[-1][None, ...], repeats=wm_num_history, axis=0)
    else:
        # gpc and iws: stack the last wm_num_history real observation frames
        return np.stack([x["image"] for x in gpc_obs_deque])  # (wm_num_history, 3, H, W)
