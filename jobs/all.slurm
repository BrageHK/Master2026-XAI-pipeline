#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=64:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --job-name="XAI-all-models"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-14

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

nvidia-smi

module purge

# Array layout: 0-4 = nnunet folds 0-4, 5-9 = swin_unetr folds 0-4, 10-14 = umamba_mtl folds 0-4
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ "$TASK_ID" -lt 5 ]; then
    MODEL="nnunet"
    FOLD=$TASK_ID
elif [ "$TASK_ID" -lt 10 ]; then
    MODEL="swin_unetr"
    FOLD=$((TASK_ID - 5))
else
    MODEL="umamba_mtl"
    FOLD=$((TASK_ID - 10))
fi

START_TIME=$(date +%Y-%m-%d_%H-%M-%S)
LOG_BASE="logs/${MODEL}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${START_TIME}"
mkdir -p "$(dirname "$LOG_BASE")"
exec > "${LOG_BASE}.txt" 2> "${LOG_BASE}.err"

echo "Running model=$MODEL fold=$FOLD"
uv run main.py --fold "$FOLD" --models "$MODEL" --methods "all" --overwrite
