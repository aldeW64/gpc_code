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

set -e

FUSION=${1:-early}
STAGE=${2:-stage1}
STAGE1_CKPT=${3:-""}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IWS_DIR="$REPO_ROOT/interactive_world_sim"
mkdir -p "$REPO_ROOT/slurm_logs"

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

if [ "$STAGE" = "stage1" ]; then
    python main.py \
        experiment=exp_latent_dyn \
        dataset=manifeel_multimodal_dataset \
        "algorithm=${ALGO}" \
        algorithm.training_stage=1 \
        algorithm.action_dim=7 \
        wandb.entity=dummy \
        wandb.mode=disabled \
        "+name=manifeel_${FUSION}_stage1_${SLURM_JOB_ID}"

elif [ "$STAGE" = "stage2" ]; then
    if [ -z "$STAGE1_CKPT" ]; then
        echo "ERROR: stage2 requires a Stage-1 checkpoint path as the third argument."
        echo "Example: sbatch slurm/train_iws_manifeel_multimodal.sh $FUSION stage2 \\"
        echo "  interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt"
        exit 1
    fi
    # Resolve to absolute path if relative
    [[ "$STAGE1_CKPT" != /* ]] && STAGE1_CKPT="$REPO_ROOT/$STAGE1_CKPT"
    echo "Stage-1 checkpoint: $STAGE1_CKPT"
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

else
    echo "ERROR: unknown stage '$STAGE'. Use 'stage1' or 'stage2'."
    exit 1
fi

echo "Done: $(date)"
