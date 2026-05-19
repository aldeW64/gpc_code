#!/usr/bin/env bash
# Download mini dataset from Hugging Face Hub.
# Usage: bash scripts/download_mini_data_hf.sh  (run from repo root)
#
# Data is organized by task:
#   Directory                    | Task
#   -----------------------------|---------------
#   data/mini/pusht/             | PushT
#   data/mini/single_grasp/      | Single Grasp
#   data/mini/bimanual_sweep/    | Bimanual Sweep
#   data/mini/bimanual_rope/     | Bimanual Rope
#
# Setup:
#   pip install huggingface_hub

set -e

python scripts/download_data_hf.py \
    --repo yixuan1999/interactive-world-sim-min-data \
    --local_dir data/mini

echo "Done. Data saved under data/mini/:"
ls data/mini
