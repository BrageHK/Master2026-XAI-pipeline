#!/bin/bash
#SBATCH --partition=short
#SBATCH --account=share-ie-idi
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --job-name="U-Mamba-XAI-metrics"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

START_TIME=$(date +%Y-%m-%d_%H-%M-%S)
JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    LOG_BASE="logs/UMamba/${JOB_ID}_${SLURM_ARRAY_TASK_ID}_${START_TIME}"
else
    LOG_BASE="logs/UMamba/${JOB_ID}_${START_TIME}"
fi
mkdir -p "$(dirname "$LOG_BASE")"
exec > "${LOG_BASE}.txt" 2> "${LOG_BASE}.err"
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge

uv run generate_xai_data.py --fold "0" --models "umamba_mtl" --methods "all"