#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=168G
#SBATCH --job-name="U-Swin-XAI-metrics"
#SBATCH --output=logs/xai/Swin_%A_%a.txt
#SBATCH --error=logs/xai/Swin_%A_%a.err
#SBATCH --array=0-4

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

module purge

uv run generate_xai_metrics-parallel.py --fold "$SLURM_ARRAY_TASK_ID"