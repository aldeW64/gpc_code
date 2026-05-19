#!/usr/bin/env python3
"""Upload dataset to Hugging Face Hub.

The local folder structure is mirrored as-is to the repo root.

Usage:
    python scripts/upload_data_hf.py \
        --repo YOUR_USERNAME/interactive-world-sim-data \
        --local_dir /path/to/data/real_aloha

First-time setup:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


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
        help="Local directory to upload (mirrored as-is to repo root)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo: https://huggingface.co/datasets/{args.repo}")

    print(f"Uploading {local_dir} → {args.repo} (repo root) ...")
    api.upload_large_folder(
        folder_path=str(local_dir),
        repo_id=args.repo,
        repo_type="dataset",
        # Chunked parallel upload with automatic resume support
    )

    print(f"\nDone. View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
