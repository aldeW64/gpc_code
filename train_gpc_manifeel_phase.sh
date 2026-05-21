#!/bin/bash
# Local debug version of slurm/train_gpc_manifeel_phase.sh
#
# Train the GPC world model on ManiFEEL data (phase 1 or phase 2).
#
# Usage (run from repo root):
#   ./train_gpc_manifeel_phase.sh phase1
#   ./train_gpc_manifeel_phase.sh phase2
#   ./train_gpc_manifeel_phase.sh phase2 \
#     world_model_train_phase_one/saved_checkpoint_manifeel/checkpoint_epoch_300/denoiser.pth
#
# Phase 1: single-step warmup (obs_horizon=4, pred_horizon=5 → seq_length=1)
# Phase 2: multi-step training (obs_horizon=4, pred_horizon=16 → seq_length=12)
#          Requires the phase-1 checkpoint; if omitted uses the default in the config YAML.

set -e

PHASE=${1:-phase1}
PHASE1_CKPT=${2:-""}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Phase:     $PHASE"
echo "Repo root: $REPO_ROOT"
echo "Time:      $(date)"

if [ "$PHASE" = "phase1" ]; then
    cd "$REPO_ROOT/world_model_train_phase_one"
    python train_manifeel.py --config configs/train_manifeel_phase_one_config.yml

elif [ "$PHASE" = "phase2" ]; then
    cd "$REPO_ROOT/world_model_train_phase_two"
    BASE_CFG="configs/train_manifeel_phase_two_config.yml"

    if [ -n "$PHASE1_CKPT" ]; then
        [[ "$PHASE1_CKPT" != /* ]] && PHASE1_CKPT="$REPO_ROOT/$PHASE1_CKPT"
        echo "Phase-1 checkpoint (override): $PHASE1_CKPT"
        TMPCONF=$(mktemp /tmp/gpc_manifeel_phase2_XXXXXX.yml)
        sed "s|phase_one_checkpoint:.*|phase_one_checkpoint: $PHASE1_CKPT|" \
            "$BASE_CFG" > "$TMPCONF"
        python train_manifeel.py --config "$TMPCONF"
        rm -f "$TMPCONF"
    else
        echo "Phase-1 checkpoint: using default from $BASE_CFG"
        python train_manifeel.py --config "$BASE_CFG"
    fi

else
    echo "ERROR: unknown phase '$PHASE'. Use 'phase1' or 'phase2'."
    exit 1
fi

echo "Done: $(date)"
