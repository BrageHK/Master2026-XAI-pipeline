#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint="a100|h100"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --job-name="nnUNet_XAI"
#SBATCH --output=logs/xai/nnUNet_XAI_%A_%a.txt
#SBATCH --error=logs/xai/nnUNet_XAI_%A_%a.err
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

module purge

uv run captum_xai.py --fold "$SLURM_ARRAY_TASK_ID" --occlusion-window "1,2,12,12" --occlusion-stride "1,1,6,6" --methods all