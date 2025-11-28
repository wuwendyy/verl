#!/usr/bin/env bash
# Shared environment, auth, CUDA/NCCL knobs, helpers for SLURM jobs
# Usage in jobs:  source /data/user_data/wendywu2/verl/scripts/cluster_shared.sh

set -euo pipefail

########################################
# ======= EDITABLE DEFAULTS ============
########################################
# Project paths (override per job if needed)
: "${REPO:=/data/user_data/wendywu2/verl}"
: "${VENV:=$REPO/.venv}"

# Data/logs (override per job if needed)
: "${DATA_DIR:=/data/user_data/wendywu2/datasets}"
: "${LOGDIR:=/data/user_data/wendywu2/logs/verl}"

# Caches (your requested lines)
export HF_HOME=/data/user_data/wendywu2/hf_cache
export TORCH_HOME="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

# CUDA/Perf toggles (safe defaults)
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  
# Enable TF32 (good speedup on A100/H100; numerically safe for training most LLMs)
export NVIDIA_TF32_OVERRIDE=1
# NCCL (single-node & common multi-node defaults)
: "${NCCL_DEBUG:=WARN}"; export NCCL_DEBUG
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   
export NCCL_NVLS_ENABLE=1
export NCCL_NET_GDR_LEVEL=2
export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE


# Hugging Face & W&B (override via env or edit here)
: "${HF_TOKEN:=}"
: "${WANDB_API_KEY:=}"
export HF_TOKEN WANDB_API_KEY

########################################
# ============ HELPERS =================
########################################
_mask() { local s="${1:-}"; [ -z "$s" ] && { echo "none"; return; }; echo "${s:0:4}****${s: -4}"; }

ensure_dirs() {
  mkdir -p "$LOGDIR" "$HF_HOME"
}

activate_venv() {
  # Activates venv and ensures repo root is importable
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV" >&2; return 1
  fi
  # Stop echoing to avoid printing secrets during auth below
  set +x
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  set -x
  cd "$REPO"
}

hf_login() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    if [ -n "${HF_TOKEN:-}" ] && [ "${HF_TOKEN}" != "__PASTE_HF_TOKEN__" ]; then
      set +e; huggingface-cli whoami -t >/dev/null 2>&1; local rc=$?; set -e
      if [ $rc -ne 0 ]; then
        set +x
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential --silent || true
        set -x
      fi
      echo "HF login OK (token=$(_mask "$HF_TOKEN"))"
    else
      echo "HF login skipped (no HF_TOKEN set)."
    fi
  else
    echo "huggingface-cli not found; skipping HF login."
  fi
}

print_job_banner() {
  # You can "module load ..." here if your cluster uses modules.
  # Example (commented): module load cuda/12.1

  echo "================= JOB BANNER ================="
  echo "Start time: $(date)"
  echo "Job ID    : ${SLURM_JOB_ID:-N/A}"
  echo "Job Name  : ${SLURM_JOB_NAME:-N/A}"
  echo "Node(s)   : ${SLURM_NODELIST:-N/A}"
  echo "Partition : ${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-N/A}}"
  echo "GPU env   : ${CUDA_VISIBLE_DEVICES:-N/A}"
  echo "=============================================="
}

# replace your existing diagnostics() with this superset (or keep your version and call nvidia-smi here)
diagnostics() {
  print_job_banner
  echo "Running nvidia-smi..."
  nvidia-smi || true
  python - <<'PY'
import torch, os, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda_rt:", torch.version.cuda)
print("cuda_devices:", torch.cuda.device_count())
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB:", "key set" if os.environ.get("WANDB_API_KEY") else "no key",
      "mode="+str(os.environ.get("WANDB_MODE")))
PY
}

init_all() {
  ensure_dirs
  activate_venv
  hf_login
  diagnostics
}

# Export functions so sub-shells can use them if needed
export -f _mask ensure_dirs activate_venv hf_login diagnostics init_all
