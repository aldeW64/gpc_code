#!/bin/bash
# Local debug version of slurm/train_gpc_manifeel_multimodal.sh
#
# Train one of the three multimodal GPC world model variants on ManiFEEL.
# Two-phase strategy:
#   Phase 1 (single-step warmup): pred_horizon=5 → seq_length=1
#   Phase 2 (multi-step):         pred_horizon=8 → seq_length=4, init from Phase 1
#
# Usage (run from repo root):
#   ./train_gpc_manifeel_multimodal.sh early  phase1
#   ./train_gpc_manifeel_multimodal.sh middle phase1
#   ./train_gpc_manifeel_multimodal.sh late   phase1
#
#   ./train_gpc_manifeel_multimodal.sh early  phase2
#   ./train_gpc_manifeel_multimodal.sh middle phase2
#   ./train_gpc_manifeel_multimodal.sh late   phase2
#
#   # Phase 2 with explicit checkpoint override:
#   ./train_gpc_manifeel_multimodal.sh early phase2 \
#     world_model_multimodal/early_fusion/saved_checkpoints_phase1/checkpoint_epoch_100/denoiser.pth

set -e

FUSION=${1:-early}
PHASE=${2:-phase1}
PHASE1_CKPT_OVERRIDE=${3:-""}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fusion:    $FUSION"
echo "Phase:     $PHASE"
echo "Repo root: $REPO_ROOT"
echo "Time:      $(date)"

case "$FUSION" in
    early|middle|late) ;;
    *)
        echo "ERROR: unknown fusion type '$FUSION'. Use 'early', 'middle', or 'late'."
        exit 1
        ;;
esac

FUSION_DIR="$REPO_ROOT/world_model_multimodal/${FUSION}_fusion"

if [ ! -d "$FUSION_DIR" ]; then
    echo "ERROR: fusion directory not found: $FUSION_DIR"
    exit 1
fi

cd "$FUSION_DIR"

if [ "$PHASE" = "phase1" ]; then
    echo "Phase 1: single-step warmup (pred_horizon=5, seq_length=1)"
    python train.py --config configs/config_phase1.yml

elif [ "$PHASE" = "phase2" ]; then
    BASE_CFG="configs/config.yml"

    if [ -n "$PHASE1_CKPT_OVERRIDE" ]; then
        [[ "$PHASE1_CKPT_OVERRIDE" != /* ]] && PHASE1_CKPT_OVERRIDE="$REPO_ROOT/$PHASE1_CKPT_OVERRIDE"
        echo "Phase 2: overriding phase_one_checkpoint → $PHASE1_CKPT_OVERRIDE"
        TMPCONF=$(mktemp /tmp/gpc_mm_phase2_XXXXXX.yml)
        sed "s|phase_one_checkpoint:.*|phase_one_checkpoint: $PHASE1_CKPT_OVERRIDE|" \
            "$BASE_CFG" > "$TMPCONF"
        python train.py --config "$TMPCONF"
        rm -f "$TMPCONF"
    else
        echo "Phase 2: using checkpoint path from $BASE_CFG"
        python train.py --config "$BASE_CFG"
    fi

else
    echo "ERROR: unknown phase '$PHASE'. Use 'phase1' or 'phase2'."
    exit 1
fi

echo "Done: $(date)"
