#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --job-name="XAI-agg-test"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

START_TIME=$(date +%Y-%m-%d_%H-%M-%S)
JOB_ID=${SLURM_JOB_ID}
LOG_BASE="logs/agg_test/${JOB_ID}_${START_TIME}"
mkdir -p "$(dirname "$LOG_BASE")"
exec > "${LOG_BASE}.txt" 2> "${LOG_BASE}.err"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_JOB_NODELIST"
echo "Started:  $START_TIME"
nvidia-smi

module purge

# Compute saliency with 3 aggregation methods on fold 0, all models.
# Appends saliency_mean / saliency_abs_sum / saliency_abs_avg fields
# to existing .npz files — existing sum data is not touched.
# Stops after 10 predicted-positive cases per model per aggregation.
uv run main.py \
    --models all \
    --fold 0 \
    --methods saliency \
    --aggregation mean abs_sum abs_avg \
    --max-cases 10 \
    --overwrite
