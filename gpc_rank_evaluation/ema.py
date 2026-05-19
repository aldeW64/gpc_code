from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass
class SimpleEMAModel:
    """
    Minimal EMA loader/copy utility.

    This is intentionally small to avoid importing `diffusers`, which can pull in
    heavyweight deps (transformers/huggingface-hub) and version constraints.
    """

    power: float = 0.75

    def __post_init__(self) -> None:
        self.shadow_params: List[torch.Tensor] = []

    @classmethod
    def from_parameters(cls, parameters: Iterable[torch.nn.Parameter], power: float = 0.75) -> "SimpleEMAModel":
        obj = cls(power=power)
        # initialize shadow with same shapes
        obj.shadow_params = [p.detach().clone().to("cpu") for p in parameters]
        return obj

    def load_state_dict(self, state_dict: dict) -> None:
        # Diffusers EMAModel stores a list of tensors under `shadow_params`
        if "shadow_params" not in state_dict:
            raise KeyError(f"EMA state_dict missing 'shadow_params'. keys={list(state_dict.keys())}")
        shadow = state_dict["shadow_params"]
        if not isinstance(shadow, list):
            raise TypeError(f"Expected shadow_params to be a list, got {type(shadow)}")
        self.shadow_params = [t.detach().to("cpu") for t in shadow]

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = list(parameters)
        if len(params) != len(self.shadow_params):
            raise ValueError(f"EMA param count mismatch: {len(params)} != {len(self.shadow_params)}")
        for p, s in zip(params, self.shadow_params):
            p.data.copy_(s.to(device=p.device, dtype=p.dtype))

