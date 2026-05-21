#!/bin/bash
# Local debug version of slurm/train_iws_manifeel_multimodal.sh
#
# Train one of the three IWS LatentWorldModel multimodal fusion variants on ManiFEEL.
# Uses both front RGB and tactile cameras.
#
# Usage (run from repo root):
#   # Stage 1 (encoder + decoder):
#   ./train_iws_manifeel_multimodal.sh early  stage1
#   ./train_iws_manifeel_multimodal.sh middle stage1
#   ./train_iws_manifeel_multimodal.sh late   stage1
#
#   # Stage 2 (dynamics) — pass the Stage-1 checkpoint path:
#   ./train_iws_manifeel_multimodal.sh early  stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt
#   ./train_iws_manifeel_multimodal.sh middle stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt
#   ./train_iws_manifeel_multimodal.sh late   stage2 \
#     interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt

set -e

FUSION=${1:-early}
STAGE=${2:-stage1}
STAGE1_CKPT=${3:-""}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IWS_DIR="$REPO_ROOT/interactive_world_sim"

echo "Fusion:    $FUSION"
echo "Stage:     $STAGE"
echo "Repo root: $REPO_ROOT"
echo "Time:      $(date)"

case "$FUSION" in
    early|middle|late) ;;
    *)
        echo "ERROR: unknown fusion type '$FUSION'. Use 'early', 'middle', or 'late'."
        exit 1
        ;;
esac

ALGO="latent_world_model_${FUSION}_fusion"

cd "$IWS_DIR"

if [ "$STAGE" = "stage1" ]; then
    python main.py \
        experiment=exp_latent_dyn \
        dataset=manifeel_multimodal_dataset \
        "algorithm=${ALGO}" \
        algorithm.training_stage=1 \
        algorithm.action_dim=7 \
        wandb.entity=dummy \
        wandb.mode=disabled \
        "+name=manifeel_${FUSION}_stage1_$$"

elif [ "$STAGE" = "stage2" ]; then
    if [ -z "$STAGE1_CKPT" ]; then
        echo "ERROR: stage2 requires a Stage-1 checkpoint path as the third argument."
        echo "Example: ./train_iws_manifeel_multimodal.sh $FUSION stage2 \\"
        echo "  interactive_world_sim/outputs/<date>/<time>/checkpoints/last.ckpt"
        exit 1
    fi
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
        "+name=manifeel_${FUSION}_stage2_$$"

else
    echo "ERROR: unknown stage '$STAGE'. Use 'stage1' or 'stage2'."
    exit 1
fi

echo "Done: $(date)"
