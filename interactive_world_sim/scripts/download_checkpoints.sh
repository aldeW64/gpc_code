#!/usr/bin/env bash
# Download pretrained checkpoints from Hugging Face Hub.
# Usage: bash scripts/download_checkpoints_hf.sh  (run from repo root)
#
# Checkpoints are organized by task and camera:
#   Directory                      | Task           | Camera
#   -------------------------------|----------------|--------
#   outputs/pusht_cam1/            | PushT          | cam1
#   outputs/single_grasp_cam0/     | Single Grasp   | cam0
#   outputs/single_grasp_cam1/     | Single Grasp   | cam1
#   outputs/bimanual_sweep_cam0/   | Bimanual Sweep | cam0
#   outputs/bimanual_sweep_cam1/   | Bimanual Sweep | cam1
#   outputs/bimanual_rope_cam0/    | Bimanual Rope  | cam0
#   outputs/bimanual_rope_cam1/    | Bimanual Rope  | cam1
#
# Each directory contains:
#   checkpoints/best.ckpt   - pretrained model weights
#   .hydra/config.yaml      - Hydra configuration used at training time
#
# Setup:
#   pip install huggingface_hub

set -e

python scripts/download_checkpoints_hf.py \
    --repo yixuan1999/interactive-world-sim-checkpoints

echo "Done. Checkpoints saved under outputs/:"
find outputs -name "best.ckpt" | sort
