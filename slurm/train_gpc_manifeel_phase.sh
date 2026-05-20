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
#          If omitted, uses the default path in the config YAML.

set -e

PHASE=${1:-phase1}
PHASE1_CKPT=${2:-""}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/slurm_logs"

source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate gpc

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
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
        # Resolve to absolute path if relative
        [[ "$PHASE1_CKPT" != /* ]] && PHASE1_CKPT="$REPO_ROOT/$PHASE1_CKPT"
        echo "Phase-1 checkpoint (override): $PHASE1_CKPT"
        # Write a temp config with the overridden checkpoint path
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
