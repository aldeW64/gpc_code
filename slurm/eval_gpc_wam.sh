#!/bin/bash
#SBATCH --job-name=eval_gpc_wam
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Evaluate using the GPC-WAM evaluation pipeline (gpc_wam_evaluation).
# Ranks candidate trajectories by -MSE(predicted_final_image, goal_image).
#
# Run from the repo root:
#   sbatch slurm/eval_gpc_wam.sh [WORLD_MODEL_TYPE] [PLANNER_MODE] [NUM_EPISODES] [EVAL_MODE]
#
# Positional arguments (all optional, order matters):
#   $1  WORLD_MODEL_TYPE  gpc (default) | wam | iws
#   $2  PLANNER_MODE      diffusion (default) | mppi
#   $3  NUM_EPISODES      number of episodes to evaluate (default: 100)
#   $4  EVAL_MODE         online_env (default) | offline_dataset
#
# Optional environment variables for checkpoint overrides:
#   WAM_CKPT_PATH    Path to a WAM (.pt) checkpoint (world_model_type=wam)
#   IWS_CKPT_PATH    Path to an IWS Lightning (.ckpt) checkpoint (world_model_type=iws)
#   IWS_CFG_PATH     Path to the Hydra config.yaml for the IWS checkpoint (auto-detected if unset)
#   OFFLINE_ZARR     Path to a zarr dataset (eval_mode=offline_dataset)
#
# Examples:
#   # GPC world model, diffusion planner, 100 online episodes (default):
#   sbatch slurm/eval_gpc_wam.sh
#
#   # IWS world model, MPPI planner, 50 episodes:
#   sbatch slurm/eval_gpc_wam.sh iws mppi 50
#
#   # Quick offline sanity check with 20 samples:
#   OFFLINE_ZARR=dataset/world_model_data/dataset_domain/all_data/domain18.zarr \
#     sbatch slurm/eval_gpc_wam.sh gpc diffusion 20 offline_dataset
#
#   # WAM backend with explicit checkpoint:
#   WAM_CKPT_PATH=wam/ckpt/pusht_finetuned/checkpoint-10000.pt \
#     sbatch slurm/eval_gpc_wam.sh wam diffusion 100

set -e

WORLD_MODEL_TYPE=${1:-gpc}
PLANNER_MODE=${2:-diffusion}
NUM_EPISODES=${3:-100}
EVAL_MODE=${4:-online_env}

REPO_ROOT="/n/holylabs/ydu_lab/Lab/pwu/Projects/gpc_code"

source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate gpc

echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURMD_NODENAME"
echo "World model type: $WORLD_MODEL_TYPE"
echo "Planner mode:     $PLANNER_MODE"
echo "Num episodes:     $NUM_EPISODES"
echo "Eval mode:        $EVAL_MODE"
echo "Repo root:        $REPO_ROOT"
echo "Time:             $(date)"

# Validate arguments
case "$WORLD_MODEL_TYPE" in
    gpc|wam|iws) ;;
    *)
        echo "ERROR: unknown world_model_type '$WORLD_MODEL_TYPE'. Use 'gpc', 'wam', or 'iws'."
        exit 1
        ;;
esac
case "$PLANNER_MODE" in
    diffusion|mppi) ;;
    *)
        echo "ERROR: unknown planner_mode '$PLANNER_MODE'. Use 'diffusion' or 'mppi'."
        exit 1
        ;;
esac
case "$EVAL_MODE" in
    online_env|offline_dataset) ;;
    *)
        echo "ERROR: unknown eval_mode '$EVAL_MODE'. Use 'online_env' or 'offline_dataset'."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Helper: find the newest Lightning .ckpt in interactive_world_sim/outputs/
# whose Hydra config has "training_stage: <stage_num>".
# Prefers last.ckpt; falls back to the newest step=*.ckpt if last.ckpt absent.
# ---------------------------------------------------------------------------
find_latest_iws_ckpt() {
    local outputs_dir="$1"
    local stage_num="$2"
    local latest_ckpt="" latest_mtime=0
    [ -d "$outputs_dir" ] || { echo ""; return; }
    # Collect all candidate checkpoints: last.ckpt and step=*.ckpt
    while IFS= read -r ckpt; do
        local cfg="${ckpt%/checkpoints/*}/.hydra/config.yaml"
        [ -f "$cfg" ] || continue
        grep -qE "training_stage:[[:space:]]*${stage_num}([^0-9]|$)" "$cfg" || continue
        local mtime
        mtime=$(stat -c %Y "$ckpt" 2>/dev/null) || continue
        if [ "$mtime" -gt "$latest_mtime" ]; then
            latest_mtime=$mtime
            latest_ckpt=$ckpt
        fi
    done < <(find "$outputs_dir" \
        \( -name "last.ckpt" -o -name "step=*.ckpt" \) -type f 2>/dev/null)
    echo "$latest_ckpt"
}

# ---------------------------------------------------------------------------
# Build the command
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"
mkdir -p slurm_logs

CMD=(
    python -m gpc_wam_evaluation.gpc_wam_evaluation
    --world_model_type "$WORLD_MODEL_TYPE"
    --planner_mode     "$PLANNER_MODE"
    --num_episodes     "$NUM_EPISODES"
    --eval_mode        "$EVAL_MODE"
)

# --- WAM checkpoint ---
if [ "$WORLD_MODEL_TYPE" = "wam" ]; then
    if [ -n "${WAM_CKPT_PATH:-}" ]; then
        [[ "$WAM_CKPT_PATH" != /* ]] && WAM_CKPT_PATH="$REPO_ROOT/$WAM_CKPT_PATH"
        echo "WAM checkpoint: $WAM_CKPT_PATH"
        CMD+=(--wam_ckpt_path "$WAM_CKPT_PATH")
    else
        echo "WAM checkpoint: auto-resolved by the evaluation script"
    fi
fi

# --- IWS checkpoint ---
if [ "$WORLD_MODEL_TYPE" = "iws" ]; then
    if [ -n "${IWS_CKPT_PATH:-}" ]; then
        [[ "$IWS_CKPT_PATH" != /* ]] && IWS_CKPT_PATH="$REPO_ROOT/$IWS_CKPT_PATH"
    else
        IWS_CKPT_PATH=$(find_latest_iws_ckpt "$REPO_ROOT/interactive_world_sim/outputs" 2)
        if [ -z "$IWS_CKPT_PATH" ]; then
            echo "ERROR: no IWS stage-2 checkpoint found. "
            echo "  Train IWS first (slurm/train_iws_manifeel_multimodal.sh <fusion> stage2)"
            echo "  or set IWS_CKPT_PATH."
            exit 1
        fi
    fi
    echo "IWS checkpoint: $IWS_CKPT_PATH"
    CMD+=(--iws_ckpt_path "$IWS_CKPT_PATH")

    if [ -n "${IWS_CFG_PATH:-}" ]; then
        [[ "$IWS_CFG_PATH" != /* ]] && IWS_CFG_PATH="$REPO_ROOT/$IWS_CFG_PATH"
        echo "IWS Hydra cfg: $IWS_CFG_PATH"
        CMD+=(--iws_cfg_path "$IWS_CFG_PATH")
    else
        echo "IWS Hydra cfg: auto-detected from checkpoint path"
    fi
fi

# --- Offline zarr path ---
if [ "$EVAL_MODE" = "offline_dataset" ]; then
    ZARR="${OFFLINE_ZARR:-dataset/world_model_data/dataset_domain/all_data/domain18.zarr}"
    [[ "$ZARR" != /* ]] && ZARR="$REPO_ROOT/$ZARR"
    if [ ! -e "$ZARR" ]; then
        echo "ERROR: offline zarr not found: $ZARR"
        echo "  Set OFFLINE_ZARR=<path> or pass eval_mode=online_env."
        exit 1
    fi
    echo "Offline zarr: $ZARR"
    CMD+=(--offline_zarr_path "$ZARR")
fi

# ---------------------------------------------------------------------------
echo ""
echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo "Done: $(date)"
