#!/bin/bash
#SBATCH --job-name=gpc_manifeel_phase
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Train the GPC world model on ManiFEEL data (phase 1 or phase 2).
#
# Run from the repo root:
#   sbatch slurm/train_gpc_manifeel_phase.sh phase1
#   sbatch slurm/train_gpc_manifeel_phase.sh phase2 \
#     world_model_train_phase_one/saved_checkpoint_manifeel/checkpoint_epoch_300/denoiser.pth
#
# Phase 1: single-step warmup (obs_horizon=4, pred_horizon=5 → seq_length=1)
# Phase 2: multi-step training (obs_horizon=4, pred_horizon=16 → seq_length=12)
#          Requires the phase-1 checkpoint path as the second argument.
#          If omitted, auto-detects the newest checkpoint in the phase-1 save dir.
#
# Auto-resume: if a previous run of the same phase was interrupted, the script
# automatically passes --resume to the training script so training continues
# from the last saved epoch rather than starting over.

set -e

PHASE=${1:-phase1}
PHASE1_CKPT=${2:-""}

REPO_ROOT="/n/holylabs/ydu_lab/Lab/pwu/Projects/gpc_code"

source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate gpc

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Phase:     $PHASE"
echo "Time:      $(date)"

# ---------------------------------------------------------------------------
# Helper: return path to the newest denoiser.pth under checkpoint_epoch_N/
# subdirectories of a given directory, or empty string if none found.
# ---------------------------------------------------------------------------
find_latest_gpc_ckpt() {
    local dir="$1"
    [ -d "$dir" ] || { echo ""; return; }
    local best_epoch=0 best_path=""
    for d in "$dir"/checkpoint_epoch_*; do
        [ -d "$d" ] || continue
        local epoch_num="${d##*_}"
        if [ "$epoch_num" -gt "$best_epoch" ] 2>/dev/null; then
            best_epoch=$epoch_num
            best_path="$d/denoiser.pth"
        fi
    done
    [ -f "$best_path" ] && echo "$best_path" || echo ""
}

# ---------------------------------------------------------------------------

if [ "$PHASE" = "phase1" ]; then
    PHASE1_SAVE_DIR="$REPO_ROOT/world_model_train_phase_one/saved_checkpoint_manifeel"
    LATEST_CKPT=$(find_latest_gpc_ckpt "$PHASE1_SAVE_DIR")

    cd "$REPO_ROOT/world_model_train_phase_one"
    if [ -n "$LATEST_CKPT" ]; then
        echo "Resuming phase-1 from: $LATEST_CKPT"
        python train_manifeel.py --config configs/train_manifeel_phase_one_config.yml \
            --resume "$LATEST_CKPT"
    else
        echo "Phase 1: starting from scratch (no prior checkpoint found)"
        python train_manifeel.py --config configs/train_manifeel_phase_one_config.yml
    fi

elif [ "$PHASE" = "phase2" ]; then
    cd "$REPO_ROOT/world_model_train_phase_two"
    BASE_CFG="configs/train_manifeel_phase_two_config.yml"

    PHASE2_SAVE_DIR="$REPO_ROOT/world_model_train_phase_two/saved_checkpoint_manifeel"
    PHASE2_RESUME=$(find_latest_gpc_ckpt "$PHASE2_SAVE_DIR")

    if [ -n "$PHASE2_RESUME" ]; then
        # An interrupted phase-2 run exists — resume it directly.
        echo "Resuming phase-2 from: $PHASE2_RESUME"
        python train_manifeel.py --config "$BASE_CFG" --resume "$PHASE2_RESUME"
    else
        # Fresh phase-2 run: need the phase-1 checkpoint to warm-start.
        if [ -z "$PHASE1_CKPT" ]; then
            PHASE1_SAVE_DIR="$REPO_ROOT/world_model_train_phase_one/saved_checkpoint_manifeel"
            PHASE1_CKPT=$(find_latest_gpc_ckpt "$PHASE1_SAVE_DIR")
            if [ -n "$PHASE1_CKPT" ]; then
                echo "Auto-detected phase-1 checkpoint: $PHASE1_CKPT"
            else
                echo "ERROR: no phase-1 checkpoint found. Run phase1 first or pass the path as \$2."
                exit 1
            fi
        fi

        [[ "$PHASE1_CKPT" != /* ]] && PHASE1_CKPT="$REPO_ROOT/$PHASE1_CKPT"
        echo "Phase-1 checkpoint: $PHASE1_CKPT"
        TMPCONF=$(mktemp /tmp/gpc_manifeel_phase2_XXXXXX.yml)
        sed "s|phase_one_checkpoint:.*|phase_one_checkpoint: $PHASE1_CKPT|" \
            "$BASE_CFG" > "$TMPCONF"
        python train_manifeel.py --config "$TMPCONF"
        rm -f "$TMPCONF"
    fi

else
    echo "ERROR: unknown phase '$PHASE'. Use 'phase1' or 'phase2'."
    exit 1
fi

echo "Done: $(date)"
