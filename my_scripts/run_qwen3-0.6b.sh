#!/bin/bash
#SBATCH --job-name=qwen3_grpo_deepscaler
#SBATCH --output=/data/user_data/wendywu2/logs/verl/build_qwen3_grpo_deepscaler_%j.out
#SBATCH --error=/data/user_data/wendywu2/logs/verl/build_qwen3_grpo_deepscaler_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:4
#SBATCH --partition=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wendywu2u2@andrew.cmu.edu

set -euo pipefail
set -x

# Single source for EVERYTHING shared
source /data/user_data/wendywu2/verl/my_scripts/shared.sh

init_all    # activates venv, logs into HF (if token), configures W&B, prints diagnostics

export VLLM_USE_V1=1

# inside /data/user_data/wendywu2/verl
bash "$REPO"/examples/grpo_trainer/run_qwen3-8b.sh
