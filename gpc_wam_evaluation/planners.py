from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateBatch:
    # (N, T, action_dim)
    actions: np.ndarray


def mppi_sample_actions(
    mean_action: np.ndarray,
    num_candidates: int,
    num_iterations: int,
    sigma: float,
    temperature: float,
    clamp_min: float,
    clamp_max: float,
    score_fn,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MPPI in action-sequence space.

    mean_action: (T, action_dim)
    score_fn: callable(actions_batch) -> scores (N,), higher is better
    Returns: (best_action_seq (T, action_dim), scores_last (N,))
    """
    assert mean_action.ndim == 2
    T, action_dim = mean_action.shape

    mu = mean_action.copy()
    scores_last = None
    for _ in range(num_iterations):
        noise = rng.normal(loc=0.0, scale=sigma, size=(num_candidates, T, action_dim)).astype(np.float32)
        candidates = mu[None, :, :] + noise
        candidates = np.clip(candidates, clamp_min, clamp_max)

        scores = score_fn(candidates)  # (N,)
        scores_last = scores

        # weights ~ exp(scores / temperature)
        s = scores.astype(np.float64)
        s = s - np.max(s)
        w = np.exp(s / max(temperature, 1e-8))
        w = w / (np.sum(w) + 1e-12)
        mu = (w[:, None, None] * candidates).sum(axis=0)

    # final selection
    best_idx = int(np.argmax(scores_last))
    return candidates[best_idx], scores_last


def diffusion_sample_actions(
    *,
    ema_nets,
    noise_scheduler,
    obs_cond: torch.Tensor,
    num_candidates: int,
    num_diffusion_iters: int,
    pred_horizon: int,
    action_dim: int,
    device: torch.device,
) -> np.ndarray:
    """
    Sample action sequences from diffusion policy.
    Returns: (N, pred_horizon, action_dim) in *normalized* action space of the policy.
    """
    # init from Gaussian
    naction = torch.randn((num_candidates, pred_horizon, action_dim), device=device)
    noise_scheduler.set_timesteps(num_diffusion_iters)
    with torch.no_grad():
        for k in noise_scheduler.timesteps:
            noise_pred = ema_nets["invariant"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond.repeat(num_candidates, 1),
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample
    return naction.detach().cpu().numpy()

