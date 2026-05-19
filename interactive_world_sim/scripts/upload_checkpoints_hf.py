#!/usr/bin/env python3
"""Upload pretrained checkpoints to Hugging Face Hub (model repo).

Uploads each task/camera subdirectory from outputs/ individually, so only
checkpoint folders (e.g. bimanual_sweep_cam0/) are included — date folders
and symlinks are automatically skipped.

Usage:
    python scripts/upload_checkpoints_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-checkpoints

    # Upload a single checkpoint directory
    python scripts/upload_checkpoints_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-checkpoints \
        --subdir bimanual_sweep_cam0

First-time setup:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# Checkpoint subdirectories to upload
CHECKPOINT_DIRS = [
    "pusht_cam1",
    "single_grasp_cam0",
    "single_grasp_cam1",
    "bimanual_sweep_cam0",
    "bimanual_sweep_cam1",
    "bimanual_rope_cam0",
    "bimanual_rope_cam1",
]


def upload_subdir(api: HfApi, repo: str, outputs_dir: Path, subdir: str):
    local_path = outputs_dir / subdir
    if not local_path.exists():
        print(f"  [skip] {subdir}/ not found locally")
        return
    print(f"  Uploading {local_path} → {repo}/{subdir}/ ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo,
        repo_type="model",
        path_in_repo=subdir,
    )
    print(f"  Done: {subdir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HF model repo ID, e.g. YOUR_USERNAME/interactive-world-sim-checkpoints",
    )
    parser.add_argument(
        "--outputs_dir",
        default="outputs",
        help="Local outputs directory (default: outputs/)",
    )
    parser.add_argument(
        "--subdir",
        default=None,
        help="Upload only this subdirectory (e.g. bimanual_sweep_cam0). Default: upload all.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs directory not found: {outputs_dir}")

    api = HfApi()

    create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo: https://huggingface.co/{args.repo}")

    subdirs = [args.subdir] if args.subdir else CHECKPOINT_DIRS
    for subdir in subdirs:
        upload_subdir(api, args.repo, outputs_dir, subdir)

    print(f"\nDone. View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
