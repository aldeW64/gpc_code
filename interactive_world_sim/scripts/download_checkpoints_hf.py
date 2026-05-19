#!/usr/bin/env python3
"""Download pretrained checkpoints from Hugging Face Hub.

Usage:
    # Download all checkpoints
    python scripts/download_checkpoints_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-checkpoints

    # Download a single checkpoint directory
    python scripts/download_checkpoints_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-checkpoints \
        --subdir bimanual_sweep_cam0

Setup:
    pip install huggingface_hub
"""

import argparse

from huggingface_hub import snapshot_download

# Checkpoint subdirectories to download
CHECKPOINT_DIRS = [
    "pusht_cam1",
    "single_grasp_cam0",
    "single_grasp_cam1",
    "bimanual_sweep_cam0",
    "bimanual_sweep_cam1",
    "bimanual_rope_cam0",
    "bimanual_rope_cam1",
]


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
        help="Local directory to download into (default: outputs/)",
    )
    parser.add_argument(
        "--subdir",
        default=None,
        help="Download only this subdirectory (e.g. bimanual_sweep_cam0). Default: download all.",
    )
    args = parser.parse_args()

    subdirs = [args.subdir] if args.subdir else CHECKPOINT_DIRS
    allow_patterns = [f"{s}/**" for s in subdirs]

    print(f"Downloading {args.repo} → {args.outputs_dir}/ ...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=args.outputs_dir,
        allow_patterns=allow_patterns,
        # Resumable: already-downloaded files are skipped
    )
    print(f"\nDone. Checkpoints saved under {args.outputs_dir}/")


if __name__ == "__main__":
    main()
