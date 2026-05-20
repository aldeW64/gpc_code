#!/bin/bash
#SBATCH --job-name=iws_manifeel
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

# Train the single-modality IWS LatentWorldModel on ManiFEEL (front RGB only).
#
# Run from interactive_world_sim/ directory:
#   sbatch scripts/train_manifeel_slurm.sh stage1
#   sbatch scripts/train_manifeel_slurm.sh stage2 outputs/<date>/<time>/checkpoints/last.ckpt

set -e

STAGE=${1:-stage1}
STAGE1_CKPT=${2:-""}

source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate iws

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Stage:  $STAGE"
echo "Time:   $(date)"

mkdir -p slurm_logs

if [ "$STAGE" = "stage1" ]; then
    python main.py \
        experiment=exp_latent_dyn \
        dataset=manifeel_dataset \
        algorithm=latent_world_model \
        algorithm.training_stage=1 \
        algorithm.action_dim=7 \
        wandb.entity=dummy \
        wandb.mode=disabled \
        "+name=manifeel_stage1_${SLURM_JOB_ID}"

elif [ "$STAGE" = "stage2" ]; then
    if [ -z "$STAGE1_CKPT" ]; then
        echo "ERROR: stage2 requires a stage1 checkpoint path as second argument"
        exit 1
    fi
    python main.py \
        experiment=exp_latent_dyn \
        dataset=manifeel_dataset \
        algorithm=latent_world_model \
        algorithm.training_stage=2 \
        algorithm.action_dim=7 \
        "algorithm.load_ae=${STAGE1_CKPT}" \
        wandb.entity=dummy \
        wandb.mode=disabled \
        "+name=manifeel_stage2_${SLURM_JOB_ID}"

else
    echo "ERROR: unknown stage '$STAGE'. Use 'stage1' or 'stage2'."
    exit 1
fi

echo "Done: $(date)"
