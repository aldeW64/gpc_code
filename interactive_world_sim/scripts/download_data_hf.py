#!/usr/bin/env python3
"""Download dataset from Hugging Face Hub.

Usage:
    # Download entire dataset
    python scripts/download_data_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-data \
        --local_dir data/full

    # Download a specific subdirectory (e.g. one task)
    python scripts/download_data_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-data \
        --local_dir data/full/bimanual_sweep \
        --repo_dir bimanual_sweep

Setup:
    pip install huggingface_hub
"""

import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HF dataset repo ID, e.g. YOUR_USERNAME/interactive-world-sim-data",
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to download into",
    )
    parser.add_argument(
        "--repo_dir",
        default=None,
        help="Subdirectory inside the repo to download (default: entire repo)",
    )
    args = parser.parse_args()

    allow_patterns = f"{args.repo_dir}/**" if args.repo_dir else None

    print(f"Downloading {args.repo}/{args.repo_dir or '(all)'} → {args.local_dir} ...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
        # Resumable: already-downloaded files are skipped
    )
    print(f"\nDone. Data saved to {args.local_dir}")


if __name__ == "__main__":
    main()
