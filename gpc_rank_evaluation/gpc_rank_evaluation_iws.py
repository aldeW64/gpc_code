"""CLI entry point for GPC-RANK evaluation with the IWS LatentWorldModel.

Usage (from repo root):
    python -m gpc_rank_evaluation.gpc_rank_evaluation_iws \
        [--config gpc_rank_evaluation/configs/gpc_rank_evaluation_iws_config.yml] \
        [--iws_ckpt_path path/to/checkpoint.ckpt]
"""

import argparse
import yaml
from pathlib import Path

from .eval_iws import eval_iws


def main() -> None:
    _default_config = str(
        Path(__file__).parent / "configs" / "gpc_rank_evaluation_iws_config.yml"
    )
    parser = argparse.ArgumentParser(
        description="GPC-RANK evaluation using the IWS LatentWorldModel."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_default_config,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--iws_ckpt_path",
        type=str,
        default=None,
        help="Override iws_world_model.ckpt_path from config.",
    )
    parser.add_argument(
        "--iws_cfg_path",
        type=str,
        default=None,
        help="Override iws_world_model.cfg_path (Hydra config.yaml) from config.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("iws_world_model", {})
    if args.iws_ckpt_path is not None:
        config["iws_world_model"]["ckpt_path"] = args.iws_ckpt_path
    if args.iws_cfg_path is not None:
        config["iws_world_model"]["cfg_path"] = args.iws_cfg_path

    eval_iws(config, config["policy_checkpoint"])


if __name__ == "__main__":
    main()
