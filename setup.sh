#!/usr/bin/env bash
set -e

# 1. Create venv (idempotent: safe even if rerun)
uv venv --python 3.12.5 .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install from requirements.txt
uv pip install -r /data/user_data/wendywu2/verl/requirements_mine.txt

python -c "import torch; print(torch.__version__)"
# 2.5.1+cu121

uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "Setup complete."
