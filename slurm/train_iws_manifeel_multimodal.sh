#!/bin/bash
#SBATCH --job-name=iws_manifeel_mm
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

# Train one of the three IWS LatentWorldModel multimodal fusion variants on ManiFEEL.
# Each variant uses BOTH front RGB and tactile cameras.
#
# Run from the repo root:
#   # Stage 1 (encoder + decoder)
#   sbatch slurm/train_iws_manifeel_multimodal.sh early  stage1
#   sbatch slurm/train_iws_manifeel_multimodal.sh middle stage1
#   sbatch slurm/train_iws_manifeel_multimodal.sh late   stage1
#
#   # Stage 2 (dynamics) — pass the Stage-1 checkpoint path
#   sbatch slurm/train_iws_manifeel_multimodal.sh early  stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt
#   sbatch slurm/train_iws_manifeel_multimodal.sh middle stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt
#   sbatch slurm/train_iws_manifeel_multimodal.sh late   stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt
#
# Fusion strategies:
#   early  — 6-ch encoder (RGB+tactile concat); predicts both modalities jointly
#   middle — dual CNN encoders + CrossModalAttention; CMDecoder outputs RGB only
#   late   — two independent pipelines; latents composed w_rgb·z_rgb + w_tac·z_tac
#
# Auto-resume:
#   Stage 1: if outputs/latest-run/checkpoints/last.ckpt exists, training resumes
#            from that checkpoint via +load=<path>.
#   Stage 2: if no Stage-1 checkpoint is given, the script searches
#            outputs/<date>/<time>/checkpoints/last.ckpt for the newest run whose
#            Hydra config contains "training_stage: 1", and uses that.

set -e

FUSION=${1:-early}
STAGE=${2:-stage1}
STAGE1_CKPT=${3:-""}

REPO_ROOT="/n/holylabs/ydu_lab/Lab/pwu/Projects/gpc_code"
IWS_DIR="$REPO_ROOT/interactive_world_sim"

source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate iws

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Fusion:    $FUSION"
echo "Stage:     $STAGE"
echo "Repo root: $REPO_ROOT"
echo "Time:      $(date)"

# Validate fusion type
case "$FUSION" in
    early|middle|late) ;;
    *)
        echo "ERROR: unknown fusion type '$FUSION'. Use 'early', 'middle', or 'late'."
        exit 1
        ;;
esac

ALGO="latent_world_model_${FUSION}_fusion"

cd "$IWS_DIR"
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Helper: find the newest last.ckpt under outputs/<date>/<time>/checkpoints/
# whose Hydra config has "training_stage: <stage_num>".
# Returns the absolute path, or empty string if not found.
# ---------------------------------------------------------------------------
find_latest_iws_ckpt() {
    local outputs_dir="$1"
    local stage_num="$2"
    local latest_ckpt="" latest_mtime=0
    [ -d "$outputs_dir" ] || { echo ""; return; }
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
    done < <(find "$outputs_dir" -path "*/checkpoints/last.ckpt" -type f 2>/dev/null)
    echo "$latest_ckpt"
}

# ---------------------------------------------------------------------------

if [ "$STAGE" = "stage1" ]; then
    # Auto-resume: if a previous stage-1 run left a last.ckpt, continue from it.
    RESUME_CKPT=$(find_latest_iws_ckpt "$IWS_DIR/outputs" 1)
    if [ -n "$RESUME_CKPT" ]; then
        echo "Resuming stage-1 from: $RESUME_CKPT"
        python main.py \
            experiment=exp_latent_dyn \
            dataset=manifeel_multimodal_dataset \
            "algorithm=${ALGO}" \
            algorithm.training_stage=1 \
            algorithm.action_dim=7 \
            wandb.entity=dummy \
            wandb.mode=disabled \
            "+load=${RESUME_CKPT}" \
            "+name=manifeel_${FUSION}_stage1_${SLURM_JOB_ID}"
    else
        echo "Stage 1: starting from scratch (no prior checkpoint found)"
        python main.py \
            experiment=exp_latent_dyn \
            dataset=manifeel_multimodal_dataset \
            "algorithm=${ALGO}" \
            algorithm.training_stage=1 \
            algorithm.action_dim=7 \
            wandb.entity=dummy \
            wandb.mode=disabled \
            "+name=manifeel_${FUSION}_stage1_${SLURM_JOB_ID}"
    fi

elif [ "$STAGE" = "stage2" ]; then
    if [ -z "$STAGE1_CKPT" ]; then
        # Auto-detect newest stage-1 checkpoint.
        STAGE1_CKPT=$(find_latest_iws_ckpt "$IWS_DIR/outputs" 1)
        if [ -n "$STAGE1_CKPT" ]; then
            echo "Auto-detected stage-1 checkpoint: $STAGE1_CKPT"
        else
            echo "ERROR: no stage-1 checkpoint found. Run stage1 first or pass the path as \$3."
            echo "Example: sbatch slurm/train_iws_manifeel_multimodal.sh $FUSION stage2 \\"
            echo "  interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt"
            exit 1
        fi
    fi
    [[ "$STAGE1_CKPT" != /* ]] && STAGE1_CKPT="$REPO_ROOT/$STAGE1_CKPT"
    echo "Stage-1 checkpoint: $STAGE1_CKPT"

    # Auto-resume: if a previous stage-2 run left a last.ckpt, continue from it.
    RESUME_CKPT=$(find_latest_iws_ckpt "$IWS_DIR/outputs" 2)
    if [ -n "$RESUME_CKPT" ]; then
        echo "Resuming stage-2 from: $RESUME_CKPT"
        python main.py \
            experiment=exp_latent_dyn \
            dataset=manifeel_multimodal_dataset \
            "algorithm=${ALGO}" \
            algorithm.training_stage=2 \
            algorithm.action_dim=7 \
            "algorithm.load_ae=${STAGE1_CKPT}" \
            wandb.entity=dummy \
            wandb.mode=disabled \
            "+load=${RESUME_CKPT}" \
            "+name=manifeel_${FUSION}_stage2_${SLURM_JOB_ID}"
    else
        python main.py \
            experiment=exp_latent_dyn \
            dataset=manifeel_multimodal_dataset \
            "algorithm=${ALGO}" \
            algorithm.training_stage=2 \
            algorithm.action_dim=7 \
            "algorithm.load_ae=${STAGE1_CKPT}" \
            wandb.entity=dummy \
            wandb.mode=disabled \
            "+name=manifeel_${FUSION}_stage2_${SLURM_JOB_ID}"
    fi

else
    echo "ERROR: unknown stage '$STAGE'. Use 'stage1' or 'stage2'."
    exit 1
fi

echo "Done: $(date)"
