import argparse
from pathlib import Path
import yaml

from .eval_wam import eval_wam


def _resolve_wam_checkpoint(user_path: str | None) -> str | None:
    """
    Resolve a WAM (CrtlWorld) checkpoint path from:
    - explicit file path (returned as-is)
    - explicit directory path (searches for known checkpoint filenames within it)
    - default repo locations, in priority order:
        1. ./wam/ckpt/pusht_finetuned/  (PushT fine-tuned, preferred)
        2. ./wam/ckpt/                   (DROID pre-trained fallback)
    """
    # WAM checkpoint filenames, highest step count first.
    # NOTE: these are CrtlWorld (.pt) checkpoints, NOT the GPC denoiser.pth files.
    known_wam_ckpt_names = [
        "checkpoint-10000.pt",
        "checkpoint-8000.pt",
        "checkpoint-6000.pt",
        "checkpoint-5000.pt",
        "checkpoint-4000.pt",
        "checkpoint-2000.pt",
        "checkpoint-latest.pt",
    ]

    if user_path:
        p = Path(user_path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            for name in known_wam_ckpt_names:
                cand = p / name
                if cand.is_file():
                    return str(cand)
        # keep user-provided path even if not found yet (will error at load time)
        return user_path

    # Default search: pusht_finetuned first, then raw pre-trained fallback
    search_dirs = [
        Path("./wam/ckpt/pusht_finetuned"),
        Path("./wam/ckpt"),
        Path("./wam_ckpt"),
    ]
    for wam_ckpt_dir in search_dirs:
        if not wam_ckpt_dir.exists():
            continue
        for name in known_wam_ckpt_names:
            cand = wam_ckpt_dir / name
            if cand.is_file():
                return str(cand)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    _default_config = str(Path(__file__).parent / "configs" / "gpc_wam_evaluation_config.yml")
    parser.add_argument(
        "--config",
        type=str,
        default=_default_config,
        help="Path to YAML config",
    )
    parser.add_argument("--wam_ckpt_path", type=str, default=None, help="Path to WAM checkpoint (.pt)")
    parser.add_argument("--planner_mode", type=str, default=None, choices=["diffusion", "mppi"])
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--eval_mode", type=str, default=None, choices=["online_env", "offline_dataset"])
    parser.add_argument("--offline_zarr_path", type=str, default=None, help="When eval_mode=offline_dataset, path to Push-T zarr")
    parser.add_argument("--world_model_type", type=str, default=None, choices=["wam", "gpc", "iws"],
                        help="World model backend: 'wam' (Ctrl-World SVD), 'gpc' (diffusion denoiser), or 'iws' (interactive_world_sim LatentWorldModel)")
    parser.add_argument("--iws_ckpt_path", type=str, default=None, help="Path to IWS Lightning checkpoint (.ckpt) when world_model_type=iws")
    parser.add_argument("--iws_cfg_path", type=str, default=None, help="Path to Hydra config.yaml for the IWS checkpoint (auto-detected from ckpt_path if omitted)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.planner_mode is not None:
        config["planner_mode"] = args.planner_mode
    if args.num_episodes is not None:
        config.setdefault("eval", {})
        config["eval"]["num_episodes"] = args.num_episodes
    if args.eval_mode is not None:
        config.setdefault("eval", {})
        config["eval"]["mode"] = args.eval_mode
    if args.offline_zarr_path is not None:
        config.setdefault("eval", {})
        if not isinstance(config["eval"].get("offline_dataset"), dict):
            config["eval"]["offline_dataset"] = {}
        config["eval"]["offline_dataset"]["zarr_path"] = args.offline_zarr_path
    if args.world_model_type is not None:
        config["world_model_type"] = args.world_model_type

    wm_type = config.get("world_model_type", "wam")
    if wm_type == "wam":
        resolved_ckpt = _resolve_wam_checkpoint(args.wam_ckpt_path)
        if resolved_ckpt is not None:
            config.setdefault("wam", {})
            config["wam"]["ckpt_path"] = resolved_ckpt
        if not config.get("wam", {}).get("ckpt_path"):
            raise ValueError(
                "world_model_type=wam but no WAM checkpoint found. "
                "Set wam.ckpt_path in YAML or pass --wam_ckpt_path."
            )
    elif wm_type == "iws":
        config.setdefault("iws_world_model", {})
        if args.iws_ckpt_path is not None:
            config["iws_world_model"]["ckpt_path"] = args.iws_ckpt_path
        if args.iws_cfg_path is not None:
            config["iws_world_model"]["cfg_path"] = args.iws_cfg_path
        if not config["iws_world_model"].get("ckpt_path"):
            raise ValueError(
                "world_model_type=iws but no IWS checkpoint found. "
                "Set iws_world_model.ckpt_path in YAML or pass --iws_ckpt_path."
            )

    print(f"[gpc_wam_eval] config={args.config}", flush=True)
    print(f"[gpc_wam_eval] world_model_type={wm_type}", flush=True)
    if wm_type == "wam":
        print(f"[gpc_wam_eval] wam_ckpt={config['wam']['ckpt_path']}", flush=True)
    elif wm_type == "iws":
        print(f"[gpc_wam_eval] iws_ckpt={config['iws_world_model']['ckpt_path']}", flush=True)
    else:
        print(f"[gpc_wam_eval] gpc_ckpt={config.get('gpc_world_model', {}).get('ckpt_path', '(from config)')}", flush=True)
    print(f"[gpc_wam_eval] output_dir={config.get('eval', {}).get('output_dir', 'gpc_wam_evaluation_output')}", flush=True)
    eval_wam(config)
    print("[gpc_wam_eval] evaluation finished", flush=True)


if __name__ == "__main__":
    main()

