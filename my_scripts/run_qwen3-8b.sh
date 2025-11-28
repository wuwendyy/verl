#!/bin/bash
#SBATCH --job-name=qwen3_grpo_deepscaler
#SBATCH --output=/data/user_data/wendywu2/logs/verl/build_qwen3_grpo_deepscaler_%j.out
#SBATCH --error=/data/user_data/wendywu2/logs/verl/build_qwen3_grpo_deepscaler_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --partition=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wendywu2u2@andrew.cmu.edu


# Load modules (adjust based on your cluster)
echo "Starting DCLM retrieval at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
nvidia-smi

set -euo pipefail
set -x

# Single source for EVERYTHING shared
source /data/user_data/wendywu2/verl/my_scripts/shared.sh

# If this job needs custom paths, override before init_all, e.g.:
# REPO=/data/user_data/wendywu2/verl
# VENV=$REPO/.venv
# DATA_DIR=/data/user_data/wendywu2/datasets/deepscaler
# export REPO VENV DATA_DIR

init_all    # activates venv, logs into HF (if token), configures W&B, prints diagnostics

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:false
# 1) Fix NCCL var rename
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 2) Remove allocator knob (or keep only max_split_size)
unset PYTORCH_CUDA_ALLOC_CONF
# or, if you really want a split size:
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

export VLLM_USE_V1=1

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE


# inside /data/user_data/wendywu2/verl
bash "$REPO"/examples/grpo_trainer/run_qwen3-8b_custom.sh
