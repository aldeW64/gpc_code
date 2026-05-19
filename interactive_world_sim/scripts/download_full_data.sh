#!/usr/bin/env bash
# Download full training dataset from Hugging Face Hub.
# Usage: bash scripts/download_full_data.sh  (run from repo root)
#
# Data is organized by task:
#   Directory                        | Task
#   ---------------------------------|--------------------
#   data/full/pusht/                 | PushT
#   data/full/single_grasp/          | Single Grasp
#   data/full/single_chain_in_box/   | Single Chain in Box
#   data/full/bimanual_sweep/        | Bimanual Sweep
#   data/full/bimanual_rope/         | Bimanual Rope
#   data/full/bimanual_box/          | Bimanual Box
#
# Setup:
#   pip install huggingface_hub

set -e

python scripts/download_data_hf.py \
    --repo yixuan1999/interactive-world-sim-data \
    --local_dir data/full

echo "Done. Data saved under data/full/:"
ls data/full
